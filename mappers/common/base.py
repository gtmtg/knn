import io

from gcloud.aio.storage import Storage
import numpy as np
from PIL import Image
import torch

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from knn.mappers import Mapper


class ResNetBackboneMapper(Mapper):
    def initialize_container(self, cfg, weights_path):
        # Create model
        shape = ShapeSpec(channels=3)
        self.model = torch.nn.Sequential(build_resnet_backbone(cfg, shape))

        # Load model weights
        checkpointer = DetectionCheckpointer(self.model, save_to_disk=False)
        checkpointer.load(weights_path)
        self.model.eval()
        torch.set_grad_enabled(False)

        # Store relevant attributes of config
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.normalize = lambda image: (image - pixel_mean) / pixel_std
        self.input_format = cfg.INPUT.FORMAT

        # Create connection pools
        self.storage_client = Storage()

    def preprocess_image(self, image):
        assert image.mode == "RGB"
        image = torch.as_tensor(np.asarray(image), dtype=torch.float32)  # -> tensor
        image = image.permute(2, 0, 1)  # HWC -> CHW
        if self.input_format == "BGR":
            image = torch.flip(image, dims=(0,))  # RGB -> BGR
        image = image.contiguous()
        return self.normalize(image)

    async def download_and_process_image(self, image_bucket, image_path, request_id):
        # Download image
        image_bytes = await self.storage_client.download(image_bucket, image_path)
        with io.BytesIO(image_bytes) as image_buffer:
            image = self.preprocess_image(Image.open(image_buffer))

        # Perform inference
        with self.profiler(request_id, "compute_time"):
            result = self.model(image.unsqueeze(dim=0))
            assert len(result) == 1
            spatial_embeddings = next(iter(result.values())).squeeze(dim=0)
            return spatial_embeddings
