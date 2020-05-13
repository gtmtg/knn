import asyncio
import io
import multiprocessing
import os

import numpy as np
from PIL import Image
import torch

from knn import utils
from knn.mappers import Mapper

from base import ResNetBackboneMapper
import config


class SpatialSearchMapper(ResNetBackboneMapper):
    async def initialize_job(self, job_args):
        return {
            **job_args,
            "template": torch.as_tensor(
                utils.base64_to_numpy(job_args["template"])
            ).unsqueeze(0),
        }

    @Mapper.SkipIfAssertionError
    async def process_element(self, input, job_id, job_args, request_id, element_index):
        image_bucket = job_args["input_bucket"]
        image_path = input
        image_name = image_path[image_path.rfind("/") + 1 : image_path.rfind(".")]

        output_bucket = job_args["output_bucket"]
        output_path = os.path.join(job_args["output_path"], job_id, image_name)

        spatial_embeddings = await self.download_and_process_image(
            image_bucket, image_path, request_id
        )
        with self.profiler(request_id, "compute_time"):
            score, score_map = self.compute_knn_score(
                spatial_embeddings,
                job_args["template"],
                job_args["n_distances_to_average"],
            )

        # Save score map
        score_map_path = os.path.join(output_path, "scores.jpg")
        await self.save_score_map(score_map, output_bucket, score_map_path)

        return {"score": score, "score_map_path": score_map_path}

    def compute_knn_score(self, embeddings, template, n_distances):
        def cosine_similarity(x1_n, x2_t_n):  # n = L2 normalized, t = transposed
            return torch.mm(x1_n, x2_t_n)

        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        embeddings_flat = torch.nn.functional.normalize(embeddings_flat, p=2, dim=0)
        scores = cosine_similarity(template, embeddings_flat).view(-1)
        score = torch.topk(scores, n_distances).values.mean().item()
        score_map = scores.view(embeddings.size(1), embeddings.size(2))
        return score, score_map

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


class SpatialSearchMasterMapper(SpatialSearchMapper):
    def initialize_container(self, input_queue, output_queue, *args, **kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        super().initialize_container(*args, **kwargs)

    def __del__(self):
        self.input_queue.put(None)

    async def process_chunk(self, chunk, job_id, job_args, request_id):
        mid = len(chunk) // 2
        master_chunk = chunk[:mid]
        slave_chunk = chunk[mid:]

        self.input_queue.put((slave_chunk, job_id, job_args, request_id))
        results = await asyncio.gather(
            *[
                self.process_element(input, job_id, job_args, request_id, i)
                for i, input in enumerate(master_chunk)
            ]
        )
        with self.profiler(request_id, "compute_time"):
            results.extend(self.output_queue.get())
            return results


def run_slave(input_queue, output_queue):
    slave = SpatialSearchMapper(
        config.RESNET_CONFIG, config.WEIGHTS_PATH, start_server=False
    )

    async def run():
        for input_chunk_args in iter(input_queue.get, None):
            output_queue.put(await slave.process_chunk(*input_chunk_args))

    asyncio.run(run())


input_queue = multiprocessing.Queue()  # type: ignore
output_queue = multiprocessing.Queue()  # type: ignore
slave = multiprocessing.Process(target=run_slave, args=(input_queue, output_queue))
slave.start()

mapper = SpatialSearchMasterMapper(
    input_queue, output_queue, config.RESNET_CONFIG, config.WEIGHTS_PATH
)
