import abc
import asyncio
import io

from gcloud.aio.storage import Storage
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.structures import ImageList

from knn.mappers import Mapper


class BatchEmbedder:
    def __init__(self, cfg, size_divisibility=16):
        self.model = torch.nn.Sequential(
            build_resnet_backbone(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        )
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1)
        self.normalize = lambda image: (image - pixel_mean) / pixel_std

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self.size_divisibility = size_divisibility

    def preprocess_image(self, image):
        if self.input_format == "BGR":
            image = image[:, :, ::-1]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # NCHW
        return image

    def preprocess_batch(self, original_images):
        images = []
        indices = []
        for image in original_images:
            if image is not None:
                indices.append(len(images))
                images.append(self.preprocess_image(image))
            else:
                indices.append(None)
        assert len(images) != 0

        image_list = ImageList.from_tensors(images, self.size_divisibility)
        image_list.tensor = self.normalize(image_list.tensor)

        masks = [torch.ones_like(image[0, ...]).unsqueeze(dim=0) for image in images]
        mask_list = ImageList.from_tensors(masks, self.size_divisibility)

        return image_list.tensor, mask_list.tensor, image_list.image_sizes, indices

    def run_on_batch(self, images):
        images, masks, sizes, indices = self.preprocess_batch(images)
        embeddings = next(iter(self.model(images).values()))
        masks = F.max_pool2d(
            masks,
            (embeddings.size(2) // masks.size(2), embeddings.size(3) // masks.size(3),),
        )
        return embeddings, masks, sizes, indices


class ResNetBackboneMapper(Mapper):
    # BASE CLASS

    @abc.abstractmethod
    async def postprocess_chunk(
        self, inputs, embeddings, masks, job_id, job_args, request_id
    ):
        pass

    # INTERNAL

    def initialize_container(self, cfg):
        self.model = BatchEmbedder(cfg)
        self.storage_client = Storage()
        torch.set_grad_enabled(False)

    async def process_chunk(self, chunk, job_id, job_args, request_id):
        # Download images
        images = await asyncio.gather(
            *[
                self.download_image(job_args["input_bucket"], input["image"])
                for input in chunk
            ]
        )

        # Compute embeddings
        with self.profiler(request_id, "compute_time"):
            try:
                embeddings, masks, sizes, indices = self.model.run_on_batch(images)
            except Exception:
                return [None] * len(chunk)

        # Gather inputs based on new indices
        gathered_inputs = [chunk[i] for i in indices if i is not None]

        # Perform task-specific postprocessing to get results
        gathered_results = await self.postprocess_chunk(
            gathered_inputs, embeddings, masks, sizes, job_id, job_args, request_id
        )

        # Scatter results back to original indices
        results = [gathered_results[i] if i is not None else None for i in indices]
        return results

    async def process_element(self, input, job_id, job_args, request_id, element_index):
        raise NotImplementedError()

    @Mapper.SkipIfError
    async def download_image(self, image_bucket, image_path):
        image_bytes = await self.storage_client.download(image_bucket, image_path)
        with io.BytesIO(image_bytes) as image_buffer:
            image = Image.open(image_buffer)
            assert image.mode == "RGB"
            return np.copy(image, order="C")
