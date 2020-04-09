import aiohttp
import asyncio
import functools
import itertools
import struct
import time

import click
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
from google.cloud.pubsub_v1.types import BatchSettings


SUBSCRIPTION_MAX_MESSAGES = 2000
REQUEST_ENDPOINT = "https://pubsub-wj4n6yaj3q-uw.a.run.app"
PROJECT_NAME = "mihir-knn"
REQUEST_QUEUE = "queries"
RESPONSE_QUEUE = "results"
SUBSCRIPTION_NAME = "pubsub-server"


def main_pubsub(n_requests, delay, interval):
    sub_client = SubscriberClient()
    pub_client = PublisherClient()

    request_queue = pub_client.topic_path(PROJECT_NAME, REQUEST_QUEUE)
    subscription_name = sub_client.subscription_path(PROJECT_NAME, SUBSCRIPTION_NAME)

    payload = struct.pack("f", delay)

    # Create initial requests
    if n_requests > 0:
        initial_pub_client = PublisherClient(BatchSettings(max_messages=n_requests))
        for i in range(n_requests):
            initial_pub_client.publish(request_queue, payload)

    start_time = time.time()
    n_received = 0
    while True:
        pull_response = sub_client.pull(
            subscription_name, SUBSCRIPTION_MAX_MESSAGES, return_immediately=True
        )
        end_time = time.time()

        ack_ids = []
        for message in pull_response.received_messages:
            if n_requests > 0:
                pub_client.publish(request_queue, payload)
            ack_ids.append(message.ack_id)

        if ack_ids:
            sub_client.acknowledge(subscription_name, ack_ids)

        n_received += len(ack_ids)
        dt = end_time - start_time
        if dt >= interval:
            print(f"Throughput: {n_received / dt}")
            start_time = end_time
            n_received = 0


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


async def request(session, delay):
    async with session.post(REQUEST_ENDPOINT, json={"delay": delay}) as response:
        return response.status == 200


def make_requests(*args, **kwargs):
    while True:
        yield request(*args, **kwargs)


@unasync
async def main_asyncio(n_requests, delay, interval):
    start_time = time.time()
    n_successful = 0
    n_total = 0
    conn = aiohttp.TCPConnector(limit=0)  # unbounded; make sure ulimit >> n_requests
    async with aiohttp.ClientSession(connector=conn) as session:
        for response in limited_as_completed(make_requests(session, delay), n_requests):
            success = await response
            n_successful += 1 if success else 0
            n_total += 1
            end_time = time.time()
            dt = end_time - start_time
            if dt >= interval:
                print(
                    f"Successful throughput: {n_successful / dt}, "
                    f"total throughput: {n_total / dt}"
                )
                start_time = end_time
                n_successful = 0
                n_total = 0


@click.command()
@click.option("--n_requests", default=50, help="Desired number of concurrent workers.")
@click.option("--delay", default=0.0, help="Desired running time of workers.")
@click.option(
    "--interval", default=10.0, help="Amount of time over which to measure throughput.",
)
@click.option("--pubsub/--no_pubsub", default=True)
def main(n_requests, delay, interval, pubsub):
    main_fn = main_pubsub if pubsub else main_asyncio
    return main_fn(n_requests, delay, interval)


if __name__ == "__main__":
    main()
