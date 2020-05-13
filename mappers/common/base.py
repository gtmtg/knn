import io
import os

from gcloud.aio.storage import Storage
import numpy as np
from PIL import Image
import torch

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from knn.mappers import Mapper


class ResNetBackboneMapper(Mapper):
    def initialize_container(
        self, cfg, weights_bucket, weights_cloud_path, weights_local_path
    ):
        # Create connection pools
        self.storage_client = Storage()

        # Load model weights
        with open(weights_local_path, "wb") as weights_file:
            weights_file.write(
                await self.storage_client.download(weights_bucket, weights_cloud_path)
            )

        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(build_resnet_backbone(cfg, shape))

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(weights_local_path)
        os.remove(weights_local_path)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Store relevant attributes of config
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.normalize = lambda image: (image - pixel_mean) / pixel_std
        self.input_format = cfg.INPUT.FORMAT

    async def download_and_process_image(self, image_bucket, image_path, request_id):
        # Download image
        image_bytes = await self.storage_client.download(image_bucket, image_path)

        # Preprocess image
        with self.profiler(request_id, "compute_time"):
            with io.BytesIO(image_bytes) as image_buffer:
                image = Image.open(image_buffer)

                # Preprocess
                assert image.mode == "RGB"
                image = torch.as_tensor(
                    np.asarray(image), dtype=torch.float32
                )  # -> tensor
                image = image.permute(2, 0, 1)  # HWC -> CHW
                if self.input_format == "BGR":
                    image = torch.flip(image, dims=(0,))  # RGB -> BGR
                image = image.contiguous()
                image = self.normalize(image)

        # Perform inference
        with self.profiler(request_id, "compute_time"):
            result = self.model(image.unsqueeze(dim=0))
            assert len(result) == 1
            spatial_embeddings = next(iter(result.values())).squeeze(dim=0)
            return spatial_embeddings
