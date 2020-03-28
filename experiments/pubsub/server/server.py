import functools
import struct
import threading
import time

import click
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
from google.cloud.pubsub_v1.types import BatchSettings


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

n_received = 0
lock = threading.Lock()


def message_callback(message, delay):
    global n_received

    with lock:
        n_received += 1

    # Create new request to replace one that just finished
    new_message = struct.pack("f", delay)
    pub_client.publish(REQUEST_QUEUE, new_message)

    message.ack()


@click.command()
@click.option("--n_requests", default=50, help="Desired number of concurrent workers.")
@click.option("--delay", default=0.0, help="Desired running time of workers.")
def main(n_requests, delay):
    global n_received

    # Create subscriber
    sub_client.create_subscription(SUBSCRIPTION_NAME, RESPONSE_QUEUE)
    sub_callback = functools.partial(message_callback, delay=delay)
    sub_future = sub_client.subscribe(SUBSCRIPTION_NAME, sub_callback)

    # Create initial requests
    initial_pub_client = PublisherClient(BatchSettings(max_messages=n_requests))
    message = struct.pack("f", delay)
    for i in range(n_requests):
        initial_pub_client.publish(REQUEST_QUEUE, message)

    try:
        while True:
            with lock:
                print(f"Throughput: {n_received}")
                n_received = 0
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sub_future.cancel()
        sub_client.delete_subscription(SUBSCRIPTION_NAME)


if __name__ == "__main__":
    main()
