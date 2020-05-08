import io
import os

import numpy as np
from PIL import Image
import torch

from knn import utils
from knn.mappers import Mapper

from base import ResNetBackboneMapper
import config


class SpatialSearchMapper(ResNetBackboneMapper):
    async def parse_args(self, args):
        return {
            **args,
            "template": utils.base64_to_numpy(args["template"]).unsqueeze(0),
        }

    @Mapper.AssertionErrorTolerant
    async def process_input(self, input, job_id, args, request_id):
        image_bucket = args["input_bucket"]
        image_path = input

        output_bucket = args["output_bucket"]
        output_path = os.path.join(args["output_path"], request_id)

        spatial_embeddings = await self.download_and_process_image(
            image_bucket, image_path, request_id
        )
        with self.profiler(request_id, "compute_time"):
            score, score_map = self.compute_knn_score(
                spatial_embeddings, args["template"], args["n_distances_to_average"]
            )

        # Save score map
        score_map_path = os.path.join(output_path, "scores.jpg")
        await self.save_score_map(score_map, output_bucket, score_map_path)

        return {"score": score, "score_map_path": score_map_path}

    async def save_score_map(self, score_map, bucket, path):
        clamped_map = torch.clamp(score_map, config.MIN_VIZ_SCORE, config.MAX_VIZ_SCORE)
        rescaled_map = (
            255
            * (clamped_map - config.MIN_VIZ_SCORE)
            / (config.MAX_VIZ_SCORE - config.MIN_VIZ_SCORE)
        )

        image = Image.fromarray(rescaled_map.numpy().astype(np.uint8))
        with io.BytesIO() as image_buffer:
            image.save(image_buffer, "jpeg")
            image_buffer.seek(0)
            await self.storage_client.upload(bucket, path, image_buffer)

    @staticmethod
    def compute_knn_score(embeddings, template, n_distances):
        def cosine_similarity(x1_n, x2_t_n):  # n = L2 normalized, t = transposed
            return torch.mm(x1_n, x2_t_n)

        with torch.no_grad():
            embeddings_flat = embeddings.view(embeddings.size(0), -1)
            embeddings_flat = torch.nn.functional.normalize(embeddings_flat, p=2, dim=0)
            scores = cosine_similarity(template, embeddings_flat).view(-1)
            score = torch.topk(scores, n_distances).values.mean().item()
            score_map = scores.view(embeddings.size(1), embeddings.size(2))
            return score, score_map


mapper = SpatialSearchMapper(config.RESNET_CONFIG, config.WEIGHTS_PATH)
