import asyncio
import collections
from dataclasses import dataclass
import heapq
import resource
import time

# TODO(mihirg): Better type annotation everywhere Dict[str, Any] is used
from typing import Optional, Iterator, List, Any, Dict, Awaitable, Tuple, Callable

import aiohttp
from runstats import Statistics

from . import utils
from . import config


# Increase maximum number of open sockets if necessary
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft = min(config.DESIRED_ULIMIT, hard)
if new_soft > soft:
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


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

    def make_result_dict(self) -> Dict[str, Any]:
        result = {
            "image_name": self.image_name,
            "score": self.score,
        }
        if self.score_map:
            result["score_map"] = self.score_map
        return result


class ImageRankingQuery:
    def __init__(
        self,
        n_concurrent_workers: int,
        n_results_to_display: int,
        handler_url: str,
        bucket_name: str,
        file_list_path: str,
        on_finished_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if not (
            config.N_CONCURRENT_WORKERS_RANGE[0]
            <= n_concurrent_workers
            <= config.N_CONCURRENT_WORKERS_RANGE[1]
        ):
            raise ValueError("Invalid number of concurrent workers")

        connector = aiohttp.TCPConnector(limit=0)
        self.session = aiohttp.ClientSession(connector=connector)

        self.n_concurrent_workers = n_concurrent_workers
        self.handler_url = handler_url

        self.bucket_name = bucket_name
        self.chunk_size = config.CHUNK_SIZE
        self.dataset = DatasetIterator(file_list_path, self.chunk_size)

        self.on_finished_callback = on_finished_callback

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

    async def set_template(self, template_request: Dict[str, Any]) -> bool:
        for i in range(config.N_TEMPLATE_ATTEMPTS):
            endpoint = f"{self.handler_url}{config.TEMPLATE_ENDPOINT}"
            async with self.session.post(endpoint, json=template_request) as response:
                if response.status == 200:
                    self.template = await response.text()
                    return True
        raise RuntimeError("Couldn't get template")

    async def start(self) -> None:
        self.query_task = asyncio.create_task(self.run_until_complete())

    async def run_until_complete(self) -> None:
        try:
            self.start_time = time.time()
            for response_tuple in utils.limited_as_completed(
                self._make_requests(), self.n_concurrent_workers
            ):
                request, result, elapsed_time = await response_tuple
                self._update_results(request, result, elapsed_time)
        except asyncio.CancelledError:
            pass
        else:
            if self.on_finished_callback is not None:
                self.on_finished_callback(self.get_results_dict())
        finally:
            await self.session.close()

    async def stop(self) -> None:
        if self.query_task is not None and not self.query_task.done():
            self.query_task.cancel()
            await self.query_task

    # REQUEST POOL

    def _make_requests(
        self,
    ) -> Iterator[Awaitable[Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]]]:
        for image_chunk in self.dataset:
            yield self._request(
                {
                    "bucket": self.bucket_name,
                    "images": image_chunk,
                    "template": self.template,
                }
            )

    async def _request(
        self, request: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], float]:
        result = None
        start_time = 0.0
        end_time = 0.0

        for i in range(config.N_QUERY_ATTEMPTS):
            start_time = time.time()
            end_time = start_time

            try:
                endpoint = f"{self.handler_url}{config.INFERENCE_ENDPOINT}"
                async with self.session.post(endpoint, json=request) as response:
                    end_time = time.time()
                    if response.status == 200:
                        result = await response.json()
                        break
            except aiohttp.ClientConnectionError:
                break

        return request, result, end_time - start_time

    # RESULT COMPILATION

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
            "running": not self.finished,
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
