import math

import torch.nn.functional as F

from knn import utils

from base import ResNetBackboneMapper
import config


class ImageEmbeddingMapper(ResNetBackboneMapper):
    async def postprocess_chunk(
        self, inputs, embeddings, masks, sizes, job_id, job_args, request_id
    ):
        with self.profiler(request_id, "compute_time"):
            results = []

            for input, size, embedding in zip(inputs, sizes, embeddings):
                x1, y1, x2, y2 = input.get("patch", (0, 0, 1, 1))
                h, w = size

                x1 = int(math.floor(x1 * w / config.RESNET_DOWNSAMPLE_FACTOR))
                y1 = int(math.floor(y1 * h / config.RESNET_DOWNSAMPLE_FACTOR))
                x2 = int(math.ceil(x2 * w / config.RESNET_DOWNSAMPLE_FACTOR))
                y2 = int(math.ceil(y2 * h / config.RESNET_DOWNSAMPLE_FACTOR))

                pooled_embedding = embedding[:, y1:y2, x1:x2].mean(dim=-1).mean(dim=-1)
                pooled_embedding = F.normalize(embedding, p=2, dim=0)
                results.append(utils.numpy_to_base64(pooled_embedding.numpy()))

            return results


mapper = ImageEmbeddingMapper(config.RESNET_CONFIG)
