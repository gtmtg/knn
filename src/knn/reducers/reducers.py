from enum import Enum
import heapq
import numpy as np

from typing import Callable, List, Optional, NamedTuple

from knn import utils
from knn.utils import JSONType

from .base import Reducer


class TopKReducer(Reducer):
    class ScoredResult(NamedTuple):
        score: float
        input: JSONType
        output: JSONType

    def __init__(
        self, k: int, extract_func: Optional[Callable[[JSONType], float]] = None
    ) -> None:
        super().__init__()
        self.k = k
        self.extract_func = extract_func or self.extract_value
        self._top_k: List[TopKReducer.ScoredResult] = []

    def handle_result(self, input: JSONType, output: JSONType) -> None:
        result = TopKReducer.ScoredResult(self.extract_func(output), input, output)
        if len(self._top_k) < self.k:
            heapq.heappush(self._top_k, result)
        elif result > self._top_k[0]:
            heapq.heapreplace(self._top_k, result)

    def extract_value(self, output: JSONType) -> float:
        assert isinstance(output, float)
        return output

    @property
    def result(self) -> List[TopKReducer.ScoredResult]:
        return list(reversed(sorted(self._top_k)))


class PoolingReducer(Reducer):
    class PoolingType(Enum):
        MAX = np.max
        AVG = np.mean

    def __init__(
        self,
        pool_type: PoolingType = PoolingType.AVG,
        extract_func: Optional[Callable[[JSONType], np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.pool_func = pool_type.value
        self.extract_func = extract_func or self.extract_value
        self._results = []  # type: List[np.ndarray]

    def handle_result(self, input: JSONType, output: JSONType) -> None:
        self._results.append(self.extract_func(output))

    def extract_value(self, output: JSONType) -> np.ndarray:
        assert isinstance(output, str)
        return utils.base64_to_numpy(output)

    @property
    def result(self) -> np.ndarray:
        return self.pool_func(np.stack(self._results), axis=0)