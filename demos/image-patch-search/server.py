import asyncio
import functools
import uuid

from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sanic import Sanic
from sanic.response import json, html, text

from knn.masters import ImageRankingQuery

import config
import utils


# Start web server
app = Sanic(__name__)
app.static("/static", "./static")
jinja = Environment(
    loader=FileSystemLoader("./templates"), autoescape=select_autoescape(["html"]),
)

current_queries = {}  # type: Dict[str, ImageRankingQuery]


@app.route("/")
async def homepage(request):
    template = jinja.get_template("index.html")
    response = template.render(
        n_concurrent_workers=config.N_CONCURRENT_WORKERS_DEFAULT,
        demo_images=config.DEMO_IMAGES,
        image_bucket=config.IMAGE_BUCKET,
        image_url_prefix=config.IMAGE_URL_PREFIX,
    )
    return html(response)


@app.route("/start", methods=["POST"])
async def start(request):
    query_id = str(uuid.uuid4())
    query = ImageRankingQuery(
        request.json["n_concurrent_workers"],
        config.N_RESULTS_TO_DISPLAY,
        config.HANDLER_URL,
        config.IMAGE_BUCKET,
        config.IMAGE_LIST_PATH,
        on_finished_callback=functools.partial(schedule_cleanup, query_id=query_id),
    )
    current_queries[query_id] = query

    await query.get_template(request.json["template"])
    await query.start()

    return json({"query_id": query_id})


@app.route("/results", methods=["GET"])
async def get_results(request):
    query_id = request.args["query_id"][0]
    query = current_queries[query_id]
    if query.finished:
        current_queries.pop(query_id)
    return json(query.get_results_dict())


@app.route("/stop", methods=["PUT"])
async def stop(request):
    query_id = request.json["query_id"]
    await current_queries.pop(query_id).stop()
    return text("", status=204)


@utils.unasync
async def schedule_cleanup(_: Any, query_id: str) -> None:
    await asyncio.sleep(config.QUERY_CLEANUP_TIME)
    current_queries.pop(query_id, None)  # don't throw error if already deleted


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
