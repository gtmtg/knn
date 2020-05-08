import asyncio
import functools
from operator import itemgetter

from typing import Dict, Iterator

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sanic import Sanic
from sanic.response import json, html, text

from knn.jobs import MapReduceJob
from knn.reducers import TopKReducer, PoolingReducer
from knn.utils import numpy_to_base64

import config


class DatasetIterator:
    def __init__(self, list_path: str) -> None:
        self._list = open(list_path, "r")
        self._total = 0
        for line in self._list:
            if not line.strip():
                break
            self._total += 1
        self._list.seek(0)

    def close(self):
        self._list.close()

    def __len__(self):
        return self._total

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        elem = self._list.readline().strip()
        if not elem:
            raise StopIteration
        return elem


# Start web server
app = Sanic(__name__)
app.static("/static", "./static")
jinja = Environment(
    loader=FileSystemLoader("./templates"), autoescape=select_autoescape(["html"]),
)

current_queries = {}  # type: Dict[str, MapReduceJob]


@app.route("/")
async def homepage(request):
    template = jinja.get_template("index.html")
    response = template.render(
        n_concurrent_workers=config.N_CONCURRENT_WORKERS_DEFAULT,
        demo_images=config.DEMO_IMAGES,
        image_bucket=config.IMAGE_BUCKET,
    )
    return html(response)


@app.route("/start", methods=["POST"])
async def start(request):
    n_mappers = request.json["n_concurrent_workers"]

    # Get template
    template_job = MapReduceJob(
        config.TEMPLATE_ENDPOINT,
        PoolingReducer(PoolingReducer.PoolingType.AVG),
        {"input_bucket": config.IMAGE_BUCKET},
        n_mappers=n_mappers,
    )
    template_request = request.json["template"]
    template_result = await template_job.run_until_complete([template_request])
    template = numpy_to_base64(template_result)

    # Run query
    query_job = MapReduceJob(
        config.QUERY_ENDPOINT,
        TopKReducer(config.N_RESULTS_TO_DISPLAY, itemgetter("score")),
        {
            "input_bucket": config.IMAGE_BUCKET,
            "output_bucket": config.OUTPUT_BUCKET,
            "template": template,
        },
        n_mappers=n_mappers,
    )
    query_id = query_job.job_id

    dataset = DatasetIterator(config.IMAGE_LIST_PATH)
    cleanup_func = functools.partial(cleanup_query, query_id=query_id, dataset=dataset)
    await query_job.start(dataset, cleanup_func)

    return json({"query_id": query_id})


@app.route("/results", methods=["GET"])
async def get_results(request):
    query_id = request.args["query_id"][0]
    query_job = current_queries[query_id]
    if query_job.finished:
        current_queries.pop(query_id)

    job_result = query_job.job_result
    job_result["result"] = [r._asdict() for r in job_result["result"]]  # make JSON-able
    return json(job_result)


@app.route("/stop", methods=["PUT"])
async def stop(request):
    query_id = request.json["query_id"]
    await current_queries.pop(query_id).stop()
    return text("", status=204)


def cleanup_query(_, query_id: str, dataset: DatasetIterator):
    dataset.close()
    asyncio.create_task(final_query_cleanup(query_id))


async def final_query_cleanup(query_id: str):
    await asyncio.sleep(config.QUERY_CLEANUP_TIME)
    current_queries.pop(query_id, None)  # don't throw error if already deleted


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
