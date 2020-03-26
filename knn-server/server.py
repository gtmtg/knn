import collections
from dataclasses import dataclass
import heapq
from multiprocessing.pool import ThreadPool
import threading
import time
import uuid

from typing import Optional, Iterator, List, Dict, Any, Union

from flask import Flask, jsonify, render_template, request
import requests

import config


class DatasetIterator:
    def __init__(self, file_list_path: str, chunk_size: str) -> None:
        self.file_list = open(file_list_path, "r")
        self.n_total = 0
        for line in self.file_list:
            if not line.strip():
                break
            self.n_total += 1
        self.file_list.seek(0)

        self.lock = threading.RLock()
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[List[str]]:
        return self

    def __next__(self) -> List[str]:
        with self.lock:
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

    def make_result_dict(self) -> Dict[str, Union[str, float]]:
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
        template: str,
        n_concurrent_workers: int,
        include_score_map: bool = True,
        n_results_to_display: int = config.N_RESULTS_TO_DISPLAY,
        file_list_path: str = config.IMAGE_LIST_PATH,
        chunk_size: int = config.CHUNK_SIZE,
    ) -> None:
        self.template = template
        self.include_score_map = include_score_map
        self.dataset = DatasetIterator(file_list_path, chunk_size)

        self.total_time = 0
        self.gcr_time = 0
        self.request_time = 0
        self.compute_time = 0
        self.n_requests = 0

        self.n_processed = 0
        self.n_skipped = 0
        self.n_chunks_per_worker = collections.defaultdict(int)

        self.results = []
        self.n_results_to_display = n_results_to_display

        self.lock = threading.RLock()
        self.pool = ThreadPool(processes=n_concurrent_workers)

        self.start_time = time.time()

    def __del__(self) -> None:
        self.pool.close()
        self.pool.join()

    def generate_requests(self) -> Iterator[Dict[str, Any]]:
        for image_chunk in self.dataset:
            yield {
                "template": self.template,
                "images": image_chunk,
                "include_score_map": self.include_score_map,
            }

    def update_results(
        self,
        chunk_request: Dict[str, Any],
        chunk_results: Optional[Dict[str, Any]],
        elapsed_time: float,
    ) -> None:
        with self.lock:
            self.n_requests += 1

            if not chunk_results:
                self.n_skipped += len(chunk_request["images"])
                return

            self.total_time += elapsed_time
            self.gcr_time += chunk_results["gcr_time"]
            self.request_time += chunk_results["request_time"]
            self.compute_time += chunk_results["compute_time"]

            self.n_processed += len(chunk_results["images"])
            self.n_skipped += len(chunk_request["images"]) - len(
                chunk_results["images"]
            )
            self.n_chunks_per_worker[chunk_results["worker_id"]] += 1

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

    def get_results_dict(self) -> Dict[str, Any]:
        with self.lock:

            def amortize(total: float) -> float:
                return total / self.n_processed if self.n_processed else 0

            return {
                "stats": {
                    "cost": self.cost,
                    "n_processed": self.n_processed,
                    "n_skipped": self.n_skipped,
                    "n_total": self.dataset.n_total,
                    "elapsed_time": time.time() - self.start_time,
                    "total_time": amortize(self.total_time),
                    "gcr_time": amortize(self.gcr_time),
                    "request_time": amortize(self.request_time),
                    "compute_time": amortize(self.compute_time),
                },
                "workers": dict(enumerate(self.n_chunks_per_worker.values())),
                "results": [
                    r.make_result_dict() for r in reversed(sorted(self.results))
                ],
            }

    @property
    def cost(self) -> float:
        with self.lock:
            return (
                0.00002400 * self.gcr_time
                + 2 * 0.00000250 * self.gcr_time
                + 0.40 / 1000000 * self.n_requests
            )

    @property
    def finished(self) -> bool:
        with self.lock:
            return self.n_processed + self.n_skipped >= self.dataset.n_total


current_queries = {}  # type: Dict[str, ImageQuery]
n_concurrent_workers = config.N_CONCURRENT_WORKERS_RANGE[1]  # default


# Start web server
app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template(
        "index.html",
        n_concurrent_workers=n_concurrent_workers,
        running=bool(current_queries),
    )


@app.route("/running")
def get_running():
    return jsonify(running=bool(current_queries))


@app.route("/toggle", methods=["POST"])
def toggle():
    global n_concurrent_workers

    # TODO(mihirg): Support multiple simultaneous queries
    if current_queries:
        current_queries.clear()
    else:
        request_json = request.get_json()
        template = None
        if request_json:
            for i in range(config.N_TEMPLATE_RETRIES):
                r = requests.post(
                    f"{config.HANDLER_URL}{config.EMBEDDING_ENDPOINT}",
                    json=request_json,
                )
                if r.status_code == 200:
                    template = r.text
                    break
        if not template:
            return "Couldn't get template", 400
        if not (
            config.N_CONCURRENT_WORKERS_RANGE[0]
            <= request_json["n_concurrent_workers"]
            <= config.N_CONCURRENT_WORKERS_RANGE[2]
        ):
            return "Invalid number of workers", 400

        query_id = uuid.uuid4()
        n_concurrent_workers = request_json["n_concurrent_workers"]
        current_queries[query_id] = ImageQuery(template, n_concurrent_workers)
        current_queries[query_id].pool.map_async(
            thread_worker, [query_id] * n_concurrent_workers
        )
    return jsonify(running=bool(current_queries))


@app.route("/results")
def get_results():
    results_dict = {}
    if current_queries:
        query = next(iter(current_queries.values()))
        if query.finished:
            current_queries.clear()
        results_dict = query.get_results_dict()
    return jsonify(**results_dict, running=bool(current_queries))


def thread_worker(query_id: str) -> bool:
    query = current_queries[query_id]
    for chunk_request in query.generate_requests():
        r = requests.post(
            f"{config.HANDLER_URL}{config.INFERENCE_ENDPOINT}", json=chunk_request
        )
        chunk_results = r.json() if r.status_code == 200 else None

        if query_id not in current_queries:
            break
        query.update_results(chunk_request, chunk_results, r.elapsed.total_seconds())

    return True
