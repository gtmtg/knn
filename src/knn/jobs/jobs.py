import asyncio
import collections
import json
import resource
import time
import uuid

import aiohttp
from runstats import Statistics

from knn import utils
from knn.utils import JSONType
from knn.reducers import Reducer

from . import defaults

from typing import (
    Optional,
    Callable,
    List,
    Dict,
    Any,
    Iterable,
)


# Increase maximum number of open sockets if necessary
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft = max(min(defaults.DESIRED_ULIMIT, hard), soft)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


class MapReduceJob:
    def __init__(
        self,
        mapper_url: str,
        reducer: Reducer,
        mapper_args: JSONType = {},
        *,
        n_mappers: int = defaults.N_MAPPERS,
        n_retries: int = defaults.N_RETRIES,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> None:
        assert n_mappers < new_soft

        self.job_id = str(uuid.uuid4())

        self.n_mappers = n_mappers
        self.n_retries = n_retries
        self.chunk_size = chunk_size
        self.mapper_url = mapper_url
        self.mapper_args = mapper_args

        self.reducer = reducer

        # Performance stats
        self._n_successful = 0
        self._n_failed = 0
        self._n_elements_per_mapper: Dict[str, int] = collections.defaultdict(int)
        self._profiling: Dict[str, Statistics] = collections.defaultdict(Statistics)

        # Will be initialized later
        self._n_total: Optional[int] = None
        self._start_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    # REQUEST LIFECYCLE

    async def start(
        self,
        iterable: Iterable[JSONType],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        async def task():
            try:
                result = await self.run_until_complete(iterable)
            except asyncio.CancelledError:
                pass
            else:
                if callback is not None:
                    callback(result)

        self.task = asyncio.create_task(task())

    async def run_until_complete(self, iterable: Iterable[JSONType]) -> Dict[str, Any]:
        assert self._start_time is None  # can't reuse Job instances
        self.start_time = time.time()

        try:
            self._n_total = len(iterable)
        except Exception:
            pass

        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            for coro in utils.limited_as_completed(
                (
                    self._request(session, list(chunk))
                    for chunk in utils.chunk(iterable, self.chunk_size)
                ),
                self.n_mappers,
            ):
                await coro

        if self._n_total is None:
            self._n_total = self._n_successful + self._n_failed
        else:
            assert self._n_total == self._n_successful + self._n_failed

        return self.result

    async def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await self.task

    # RESULT GETTERS

    @property
    def result(self) -> Any:
        return self.reducer.result

    @property
    def job_result(self) -> Dict[str, Any]:
        elapsed_time = (time.time() - self.start_time) if self.start_time else 0.0

        performance = {
            "profiling": {k: v.mean() for k, v in self._profiling.items()},
            "mapper_utilization": dict(enumerate(self._n_elements_per_mapper.values())),
        }

        progress = {
            "finished": self.finished,
            "n_processed": self._n_successful,
            "n_skipped": self._n_failed,
            "elapsed_time": elapsed_time,
        }
        if self._n_total is not None:
            progress["n_total"] = self._n_total

        return {
            "performance": performance,
            "progress": progress,
            "result": self.result,
        }

    @property
    def finished(self) -> bool:
        return self._n_total == self._n_successful + self._n_failed

    # INTERNAL

    def _construct_request(self, chunk: List[JSONType]) -> JSONType:
        return {
            "job_id": self.job_id,
            "job_args": self.mapper_args,
            "inputs": chunk,
        }

    async def _request(self, session: aiohttp.ClientSession, chunk: List[JSONType]):
        request = self._construct_request(chunk)

        for i in range(self.n_retries):
            remaining_indices = set(range(len(chunk)))

            try:
                async with session.post(self.mapper_url, json=request) as response:
                    buffer = b""
                    async for data, end_of_http_chunk in response.content.iter_chunks():
                        buffer += data
                        if end_of_http_chunk:
                            text = buffer.decode()
                            if "HTTP/1.1" in text and ("500" in text or "429" in text):
                                raise RuntimeError(text)
                            parts = text.split("\r\n")
                            buffer = b""
                            for part in parts:
                                try:
                                    payload = json.loads(part)
                                    assert isinstance(payload, dict)
                                except (json.decoder.JSONDecodeError, AssertionError):
                                    pass
                                else:
                                    i = payload["index"]
                                    self._handle_result(chunk[i], payload)
                                    remaining_indices.remove(i)
                                    break
            except asyncio.CancelledError:
                pass
            except Exception:
                print(0)
            else:
                print(len(remaining_indices))

            chunk = [chunk[i] for i in remaining_indices]

        for input in chunk:
            self._handle_result(input, None)

    def _handle_result(self, input: JSONType, result: Optional[JSONType]):
        if not result or result.get("output") is None:
            self._n_failed += 1
            return

        self._n_successful += 1
        self._n_elements_per_mapper[result["worker_id"]] += 1
        self.reducer.handle_result(input, result["output"])
