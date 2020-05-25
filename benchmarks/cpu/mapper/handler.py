import time
from knn.mappers import Mapper


class BenchmarkCPUMapper(Mapper):
    async def process_element(self, input, job_id, job_args, request_id, element_index):
        start_time = time.time()
        n = 0
        while time.time() - start_time < job_args["runtime"]:
            n += 1
        return n


mapper = BenchmarkCPUMapper()
