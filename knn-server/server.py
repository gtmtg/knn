import asyncio
import collections
from dataclasses import dataclass
import heapq
import time
import uuid

from typing import Optional, Iterator, List, Any, Dict, Awaitable, Tuple

import aiohttp
from jinja2 import Environment, FileSystemLoader, select_autoescape
from runstats import Statistics
from sanic import Sanic
from sanic.response import json, html

import config
import utils


# Start web server
app = Sanic(__name__)
app.static("/static", "./static")
jinja = Environment(
    loader=FileSystemLoader("./templates"), autoescape=select_autoescape(["html"]),
)

current_queries = {}  # type: Dict[str, ImageQuery]
n_concurrent_workers_cached = config.N_CONCURRENT_WORKERS_RANGE[1]  # default


class DatasetIterator:
    def __init__(self, file_list_path: str, chunk_size: int) -> None:
        self.file_list = open(file_list_path, "r")
        self.n_total = 0
        for line in self.file_list:
            if not line.strip():
                break
            self.n_total += 1
        self.file_list.seek(0)

        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[List[str]]:
        return self

    def __next__(self) -> List[str]:
        first_file = self.file_list.readline().strip()
        if not first_file:
            raise StopIteration

        files = [first_file]
        while len(files) < self.chunk_size:
            next_file = self.file_list.readline().strip()
            if next_file:
                files.append(next_file)
            else:
                break
        return files


@dataclass(order=True)
class ImageResult:
    score: float
    image_name: str
    score_map: Optional[str] = None

    # TODO(mihirg): Better type annotation
    def make_result_dict(self) -> Dict[str, Any]:
        result = {
            "image_name": self.image_name,
            "image_url": config.IMAGE_URL_FORMAT.format(self.image_name),
            "score": self.score,
        }
        if self.score_map:
            result["score_map_url"] = f"data:image/jpeg;base64,{self.score_map}"
        return result


class ImageQuery:
    def __init__(
        self,
        n_concurrent_workers: int,
        include_score_map: bool = True,
        n_results_to_display: int = config.N_RESULTS_TO_DISPLAY,
        file_list_path: str = config.IMAGE_LIST_PATH,
        chunk_size: int = config.CHUNK_SIZE,
    ) -> None:
        connector = aiohttp.TCPConnector(limit=0)  # no limit, make sure ulimit is high!
        self.session = aiohttp.ClientSession(connector=connector)
        self.n_concurrent_workers = n_concurrent_workers

        self.include_score_map = include_score_map
        self.chunk_size = chunk_size
        self.dataset = DatasetIterator(file_list_path, chunk_size)

        self.total_time = Statistics()
        self.gcr_time = Statistics()
        self.request_time = Statistics()
        self.compute_time = Statistics()

        self.n_processed = 0
        self.n_skipped = 0
        self.n_requests = 0
        self.n_requests_per_worker: Dict[str, int] = collections.defaultdict(int)

        self.results: List[ImageResult] = []
        self.n_results_to_display = n_results_to_display

        # Will be initialized later
        self.template: Optional[str] = None
        self.start_time: Optional[float] = None
        self.query_task: Optional[asyncio.Task] = None

    # TODO(mihirg): Better type annotation
    async def set_template(self, template_request: Dict[str, Any]) -> bool:
        for i in range(config.N_TEMPLATE_ATTEMPTS):
            endpoint = f"{config.HANDLER_URL}{config.TEMPLATE_ENDPOINT}"
            async with self.session.post(endpoint, json=template_request) as response:
                if response.status == 200:
                    self.template = await response.text()
                    return True
        raise RuntimeError("Couldn't get template")

    def start(self) -> None:
        self.query_task = asyncio.create_task(self._run_query())

    @utils.unasync_eventually
    async def stop(self) -> None:
        if self.query_task is not None:
            self.query_task.cancel()
            try:
                await self.query_task
            except asyncio.CancelledError:
                pass
        await self.session.close()

    # REQUEST POOL

    # TODO(mihirg): Better type annotation
    async def _request(
        self, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]:
        result = None
        start_time = time.time()
        end_time = start_time

        try:
            endpoint = f"{config.HANDLER_URL}{config.INFERENCE_ENDPOINT}"
            async with self.session.post(endpoint, json=request) as response:
                end_time = time.time()
                if response.status == 200:
                    result = await response.json()
        except aiohttp.ClientConnectionError:
            pass
        return request, result, end_time - start_time

    # TODO(mihirg): Better type annotation
    def _make_requests(
        self,
    ) -> Iterator[Awaitable[Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]]]:
        for image_chunk in self.dataset:
            yield self._request(
                {
                    "template": self.template,
                    "images": image_chunk,
                    "include_score_map": self.include_score_map,
                }
            )

    async def _run_query(self) -> None:
        self.start_time = time.time()
        for response_tuple in utils.limited_as_completed(
            self._make_requests(), self.n_concurrent_workers
        ):
            request, result, elapsed_time = await response_tuple
            self._update_results(request, result, elapsed_time)

    # RESULT COMPILATION

    # TODO(mihirg): Better type annotation
    def _update_results(
        self,
        chunk_request: Dict[str, Any],
        chunk_results: Optional[Dict[str, Any]],
        elapsed_time: float,
    ) -> None:
        self.n_requests += 1

        if not chunk_results:
            self.n_skipped += len(chunk_request["images"])
            return

        self.total_time.push(elapsed_time)
        self.gcr_time.push(chunk_results["gcr_time"])
        self.request_time.push(chunk_results["request_time"])
        self.compute_time.push(chunk_results["compute_time"])

        self.n_processed += len(chunk_results["images"])
        self.n_skipped += len(chunk_request["images"]) - len(chunk_results["images"])
        self.n_requests_per_worker[chunk_results["worker_id"]] += 1

        for image_name, result_dict in chunk_results["images"].items():
            result = ImageResult(
                image_name=image_name,
                score=result_dict["score"],
                score_map=result_dict.get("score_map"),
            )
            if len(self.results) < self.n_results_to_display:
                heapq.heappush(self.results, result)
            elif result > self.results[0]:
                heapq.heapreplace(self.results, result)

    # TODO(mihirg): Better type annotation
    def get_results_dict(self) -> Dict[str, Any]:
        elapsed_time = (time.time() - self.start_time) if self.start_time else 0
        return {
            "stats": {
                "cost": self.cost,
                "n_processed": self.n_processed,
                "n_skipped": self.n_skipped,
                "n_total": self.dataset.n_total,
                "elapsed_time": elapsed_time,
                "total_time": self.total_time.mean(),
                "gcr_time": self.gcr_time.mean(),
                "request_time": self.request_time.mean(),
                "compute_time": self.compute_time.mean(),
                "chunk_size": self.chunk_size,
            },
            "workers": dict(enumerate(self.n_requests_per_worker.values())),
            "results": [r.make_result_dict() for r in reversed(sorted(self.results))],
        }

    @property
    def cost(self) -> float:
        total_gcr_time = self.gcr_time.mean() * len(self.gcr_time)
        return (
            0.00002400 * total_gcr_time
            + 2 * 0.00000250 * total_gcr_time
            + 0.40 / 1000000 * self.n_requests
        )

    @property
    def finished(self) -> bool:
        return self.n_processed + self.n_skipped >= self.dataset.n_total


@app.route("/")
async def homepage(request):
    template = jinja.get_template("index.html")
    response = template.render(
        n_concurrent_workers=n_concurrent_workers_cached, running=bool(current_queries)
    )
    return html(response)


@app.route("/running")
async def get_running(request):
    return json({"running": bool(current_queries)})


@app.route("/toggle", methods=["POST"])
async def toggle(request):
    global n_concurrent_workers_cached

    # TODO(mihirg): Support multiple simultaneous queries
    if current_queries:
        clear_queries()
    else:
        # Validate request
        assert (
            config.N_CONCURRENT_WORKERS_RANGE[0]
            <= request.json["n_concurrent_workers"]
            <= config.N_CONCURRENT_WORKERS_RANGE[2]
        )
        n_concurrent_workers_cached = request.json["n_concurrent_workers"]

        query = ImageQuery(n_concurrent_workers_cached)
        current_queries[uuid.uuid4()] = query
        await query.set_template(request.json)
        query.start()

    return json({"running": bool(current_queries)})


@app.route("/results")
async def get_results(request):
    results_dict = {}
    if current_queries:
        query = next(iter(current_queries.values()))
        if query.finished:
            clear_queries()
        results_dict = query.get_results_dict()

    results_dict["running"] = bool(current_queries)
    return json(results_dict)


def clear_queries():
    query_ids = list(current_queries.keys())
    for query_id in query_ids:
        current_queries.pop(query_id).stop()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
