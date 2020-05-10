import aiohttp
import asyncio
import functools
import itertools

# import struct
import time

import click

# from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
# from google.cloud.pubsub_v1.types import BatchSettings


SUBSCRIPTION_MAX_MESSAGES = 2000
# REQUEST_ENDPOINT = ""
# REQUEST_ENDPOINT = "https://knn-wj4n6yaj3q-uw.a.run.app/delay"
# REQUEST_ENDPOINT = "https://large-container-sleep-microbenchmark-central1-wj4n6yaj3q-uc.a.run.app/delay"  # noqa
# REQUEST_ENDPOINT = "https://small-container-sleep-microbenchmark-central1-wj4n6yaj3q-uc.a.run.app"  # noqa
PROJECT_NAME = "mihir-knn"
REQUEST_QUEUE = "queries"
RESPONSE_QUEUE = "results"
SUBSCRIPTION_NAME = "pubsub-server"


# WEST 1
# ENDPOINTS = {
#     (
#         False,
#         500,
#     ): "https://small-container-500-worker-sleep-microbenchmark-w-wj4n6yaj3q-uw.a.run.app",
#     (
#         False,
#         1000,
#     ): "https://small-container-1k-worker-sleep-microbenchmark-we-wj4n6yaj3q-uw.a.run.app",
#     (
#         True,
#         500,
#     ): "https://large-container-500-worker-sleep-microbenchmark-w-wj4n6yaj3q-uw.a.run.app/delay",
#     (
#         True,
#         1000,
#     ): "https://large-container-1k-worker-sleep-microbenchmark-we-wj4n6yaj3q-uw.a.run.app/delay",
# }

# CENTRAL 1
ENDPOINTS = {
    (
        False,
        500,
    ): "https://small-container-500-worker-sleep-microbenchmark-c-wj4n6yaj3q-uc.a.run.app",
    (
        False,
        1000,
    ): "https://small-container-1k-worker-sleep-microbenchmark-ce-wj4n6yaj3q-uc.a.run.app",
    (
        True,
        500,
    ): "https://large-container-500-worker-sleep-microbenchmark-c-wj4n6yaj3q-uc.a.run.app/delay",
    (
        True,
        1000,
    ): "https://large-container-1k-worker-sleep-microbenchmark-ce-wj4n6yaj3q-uc.a.run.app/delay",
}

# def main_pubsub(n_requests, delay, interval, end_after):
#     sub_client = SubscriberClient()
#     pub_client = PublisherClient()

#     request_queue = pub_client.topic_path(PROJECT_NAME, REQUEST_QUEUE)
#     subscription_name = sub_client.subscription_path(PROJECT_NAME, SUBSCRIPTION_NAME)

#     payload = struct.pack("f", delay)

#     # Create initial requests
#     if n_requests > 0:
#         initial_pub_client = PublisherClient(BatchSettings(max_messages=n_requests))
#         for i in range(n_requests):
#             initial_pub_client.publish(request_queue, payload)

#     start_time = time.time()
#     n_received = 0
#     n_finished = 0
#     while True:
#         pull_response = sub_client.pull(
#             subscription_name, SUBSCRIPTION_MAX_MESSAGES, return_immediately=True
#         )
#         end_time = time.time()

#         ack_ids = []
#         for message in pull_response.received_messages:
#             if n_requests > 0:
#                 pub_client.publish(request_queue, payload)
#             ack_ids.append(message.ack_id)

#         if ack_ids:
#             sub_client.acknowledge(subscription_name, ack_ids)

#         n_received += len(ack_ids)
#         dt = end_time - start_time
#         if dt >= interval:
#             print(f"Throughput: {n_received / dt}")

#             n_finished += 1
#             if n_finished == end_after:
#                 break

#             start_time = end_time
#             n_received = 0


def limited_as_completed(coros, limit):
    # From https://github.com/andybalaam/asyncioplus/blob/master/asyncioplus/limited_as_completed.py  # noqa
    futures = [asyncio.ensure_future(c) for c in itertools.islice(coros, 0, limit)]

    async def first_to_finish():
        while True:
            await asyncio.sleep(0)
            for f in futures:
                if f.done():
                    futures.remove(f)
                    try:
                        newf = next(coros)
                        futures.append(asyncio.ensure_future(newf))
                    except StopIteration:
                        pass
                    return f.result()

    while len(futures) > 0:
        yield first_to_finish()


def unasync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        asyncio.run(f(*args, **kwargs))

    return wrapper


async def request(session, delay, endpoint):
    async with session.post(endpoint, json={"delay": delay}) as response:
        return response.status == 200


def make_requests(*args, **kwargs):
    while True:
        yield request(*args, **kwargs)


@unasync
async def main_asyncio(n_requests, delay, interval, end_after, large):
    n_successful = [0]
    n_total = [0]
    conn = aiohttp.TCPConnector(limit=0)  # unbounded; make sure ulimit >> n_requests

    endpoint = ENDPOINTS[(large, n_requests)]
    first_successful = None

    async def benchmark():
        n_finished = 0
        while True:
            await asyncio.sleep(interval)
            print(f"{n_successful[0]}, {n_total[0]}")
            n_successful[0] = 0
            n_total[0] = 0

            n_finished += 1
            if n_finished == end_after:
                break

    benchmark_task = asyncio.create_task(benchmark())
    start_time = time.time()

    async with aiohttp.ClientSession(connector=conn) as session:
        for response in limited_as_completed(
            make_requests(session, delay, endpoint), n_requests
        ):
            if benchmark_task.done():
                break
            success = await response
            if not first_successful and success:
                first_successful = time.time() - start_time

            n_successful[0] += 1 if success else 0
            n_total[0] += 1

    print(f"First successful: {first_successful}")


@click.command()
@click.option("--n_requests", default=50, help="Desired number of concurrent workers.")
@click.option("--delay", default=0.0, help="Desired running time of workers.")
@click.option(
    "--interval", default=10.0, help="Amount of time over which to measure throughput.",
)
@click.option(
    "--end_after", default=0, help="Number of intervals to run for (0 = unlimited)."
)
@click.option("--large/--small", default=False)
def main(n_requests, delay, interval, end_after, large):
    return main_asyncio(n_requests, delay, interval, end_after, large)
    # main_fn = main_pubsub if pubsub else main_asyncio
    # return main_fn(n_requests, delay, interval, end_after)


if __name__ == "__main__":
    main()
