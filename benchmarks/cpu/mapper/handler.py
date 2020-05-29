import asyncio
import multiprocessing
import os
import time
from knn.mappers import Mapper


class BenchmarkCPUMapper(Mapper):
    async def process_element(self, input, job_id, job_args, request_id, element_index):
        start_time = time.time()
        n = 0
        while time.time() - start_time < job_args["runtime"]:
            n += 1
        return n


class BenchmarkCPUMasterMapper(BenchmarkCPUMapper):
    def initialize_container(self, input_queue, output_queue, *args, **kwargs):
        self.input_queue = input_queue
        self.output_queue = output_queue
        super().initialize_container(*args, **kwargs)

    def __del__(self):
        self.input_queue.put(None)

    async def process_chunk(self, chunk, job_id, job_args, request_id):
        self.input_queue.put((chunk, job_id, job_args, request_id))
        master_results = await asyncio.gather(
            *[
                self.process_element(input, job_id, job_args, request_id, i)
                for i, input in enumerate(chunk)
            ]
        )
        with self.profiler(request_id, "compute_time"):
            slave_results = output_queue.get()
            results = list(map(sum, zip(master_results, slave_results)))
            return results


def run_slave(input_queue, output_queue):
    async def run():
        slave = BenchmarkCPUMapper(start_server=False)

        for input_chunk_args in iter(input_queue.get, None):
            output_queue.put(await slave.process_chunk(*input_chunk_args))

    asyncio.run(run())


if os.cpu_count() == 1:
    mapper = BenchmarkCPUMapper()
else:  # treat as 2
    input_queue = multiprocessing.Queue()  # type: ignore
    output_queue = multiprocessing.Queue()  # type: ignore
    slave = multiprocessing.Process(target=run_slave, args=(input_queue, output_queue))
    slave.start()

    mapper = BenchmarkCPUMasterMapper(input_queue, output_queue)
