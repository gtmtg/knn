import asyncio
import concurrent
import functools
import struct

import click
from gcloud.aio.pubsub import SubscriberClient
from google.cloud.pubsub_v1 import PublisherClient
from google.cloud.pubsub_v1.types import BatchSettings

from async_publish import AsyncioBatch


N_CONCURRENT_REQUESTS = 50
DELAY = 0.0

REQUEST_QUEUE = "projects/{project_id}/topics/{topic}".format(
    project_id="mihir-knn", topic="queries"
)
RESPONSE_QUEUE = "projects/{project_id}/topics/{topic}".format(
    project_id="mihir-knn", topic="results"
)
SUBSCRIPTION_NAME = "projects/{project_id}/subscriptions/{sub}".format(
    project_id="mihir-knn", sub="pubsub-server"
)

PUBLISH_MAX_LATENCY = 0.1


sub_client = SubscriberClient()

# https://github.com/mozilla/gcp-ingestion/blob/24c1cea1bdcb69f6f38a86f73a8ae28978cdfecf/ingestion-edge/ingestion_edge/publish.py#L87  # noqa
pub_client = PublisherClient(BatchSettings(max_latency=PUBLISH_MAX_LATENCY))
pub_client._batch_class = AsyncioBatch

n_received = 0


async def message_callback(message, delay):
    global n_received
    n_received += 1

    # Create new request to replace one that just finished
    new_message = struct.pack("f", delay)
    await pub_client.publish(REQUEST_QUEUE, new_message)

    message.ack()


def unasync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        asyncio.run(f(*args, **kwargs))

    return wrapper


@click.command()
@click.option("--n_requests", default=50, help="Desired number of concurrent workers.")
@click.option("--delay", default=0.0, help="Desired running time of workers.")
@unasync
async def main(n_requests, delay):
    global n_received

    # Create subscriber
    sub_client.create_subscription(SUBSCRIPTION_NAME, RESPONSE_QUEUE)
    sub_callback = functools.partial(message_callback, delay=delay)
    sub_keepalive = sub_client.subscribe(SUBSCRIPTION_NAME, sub_callback)

    # Create initial requests
    initial_pub_client = PublisherClient(
        BatchSettings(max_messages=N_CONCURRENT_REQUESTS)
    )
    initial_pub_client._batch_class = AsyncioBatch
    message = struct.pack("f", delay)
    for i in range(n_requests):
        await initial_pub_client.publish(REQUEST_QUEUE, message)

    try:
        # Print throughput every second
        while True:
            print(f"Throughput: {n_received}")
            n_received = 0
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, concurrent.futures.CancelledError):
        pass
    finally:
        # Cancel subscription to prevent more tasks from being leased
        if not sub_keepalive.cancelled():
            sub_keepalive.cancel()

        # Cancel existing tasks in subscriber client's event loop
        for task in asyncio.Task.all_tasks(loop=sub_client.loop):
            task.cancel()


if __name__ == "__main__":
    main()
