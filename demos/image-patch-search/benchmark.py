import asyncio
import click
import json

from knn.jobs import MapReduceJob
from knn.reducers import Reducer
from knn.utils import FileListIterator, unasync

import config

# Some random encoded patch embedding
TEMPLATE = "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"


class EmptyReducer(Reducer):
    def handle_result(self, input, output):
        pass

    @property
    def result(self):
        return None


@click.command()
@click.option("-m", "--mapper", default=config.QUERY_ENDPOINT)
@click.option("-w", "--workers", default=1000)
@click.option("-i", "--interval", default=5)
@click.argument("output", type=click.File("w"))
@unasync
async def main(mapper, workers, interval, output):
    query_job = MapReduceJob(
        mapper,
        EmptyReducer(),
        {
            "input_bucket": config.IMAGE_BUCKET,
            "output_bucket": config.OUTPUT_BUCKET,
            "output_path": config.OUTPUT_PATH,
            "n_distances_to_average": config.N_DISTANCES_TO_AVERAGE,
            "template": TEMPLATE,
        },
        n_mappers=workers,
        n_retries=1,
    )

    dataset = FileListIterator(config.IMAGE_LIST_PATH)
    await query_job.start(dataset, dataset.close)

    results = []
    try:
        while not query_job.finished:
            await asyncio.sleep(interval)
            results.append(query_job.job_result)
    except KeyboardInterrupt:
        pass
    finally:
        json.dump(results, output)


if __name__ == "__main__":
    main()
