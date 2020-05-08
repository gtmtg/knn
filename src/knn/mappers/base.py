import abc
import asyncio
import collections
from dataclasses import dataclass
import functools
import time
import uuid

from typing import List, Dict, Any

from sanic import Sanic
from sanic.response import json

from knn.utils import JSONType


@dataclass
class Profiler:
    request_id: str
    category: str
    results_dict: Dict[str, Dict[str, float]]
    additional: float = 0.0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.results_dict[self.request_id][self.category] += (
            time.time() - self.start_time + self.additional
        )


class Mapper(abc.ABC):
    # INTERNAL

    def __init__(self, *args, **kwargs):
        init_start_time = time.time()

        self.worker_id = str(uuid.uuid4())
        self.job_args: Dict[str, Any] = {}
        self.request_profiling: Dict[str, Dict[str, float]] = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )
        self.profiler = functools.partial(Profiler, results_dict=self.request_profiling)

        self.initialize_container(*args, **kwargs)

        self._server = Sanic(self.worker_id)
        self._server.add_route(self._handle_request, "/", methods=["POST"])
        self._server.add_route(self._sleep, "/sleep", methods=["POST"])  # benchmarking

        self._init_time = time.time() - init_start_time

    async def __call__(self, *args, **kwargs):
        return await self._server(*args, **kwargs)

    async def _sleep(self, request):
        delay = float(request.json["delay"])
        asyncio.sleep(delay)
        return json(request.json)

    async def _handle_request(self, request):
        init_time = self._init_time
        self._init_time = 0.0

        request_id = str(uuid.uuid4())
        with self.profiler(request_id, "billed_time", additional=init_time):
            with self.profiler(request_id, "request_time"):
                job_id = request.json["job_id"]
                args = self.job_args.setdefault(
                    job_id, await self.parse_args(request.json["args"])
                )
                outputs = await self.process_batch(
                    request.json["inputs"], job_id, args, request_id
                )

        return json(
            {
                "worker_id": self.worker_id,
                "profiling": self.request_profiling.pop(request_id),
                "outputs": outputs,
            }
        )

    # BASE CLASS

    def initialize_container(self, *args, **kwargs) -> None:
        pass

    async def parse_args(self, args: JSONType) -> Any:
        return args

    async def process_batch(
        self, batch: List[JSONType], job_id: str, args: Any, request_id: str
    ) -> List[JSONType]:
        return await asyncio.gather(
            *[self.process_input(input, job_id, args, request_id) for input in batch]
        )

    @abc.abstractmethod
    async def process_input(
        self, input: JSONType, job_id: str, args: Any, request_id: str
    ) -> JSONType:
        pass

    # DECORATORS

    @staticmethod
    def AssertionErrorTolerant(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except AssertionError:
                pass
            return None
