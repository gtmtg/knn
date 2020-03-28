import struct
import time

import click
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
from google.cloud.pubsub_v1.types import BatchSettings


MIN_THROUGHPUT_TIME = 10.0
SUBSCRIPTION_MAX_MESSAGES = 2000


sub_client = SubscriberClient()
pub_client = PublisherClient()


REQUEST_QUEUE = pub_client.topic_path("mihir-knn", "queries")
RESPONSE_QUEUE = sub_client.topic_path("mihir-knn", "results")
SUBSCRIPTION_NAME = sub_client.subscription_path("mihir-knn", "pubsub-server")


# n_received = 0
# lock = threading.Lock()


# def message_callback(message, delay):
#     global n_received

#     with lock:
#         n_received += 1

#     # Create new request to replace one that just finished
#     new_message = struct.pack("f", delay)
#

#     message.ack()


@click.command()
@click.option("--n_requests", default=50, help="Desired number of concurrent workers.")
@click.option("--delay", default=0.0, help="Desired running time of workers.")
def main(n_requests, delay):
    payload = struct.pack("f", delay)

    # Create initial requests
    initial_pub_client = PublisherClient(BatchSettings(max_messages=n_requests))
    for i in range(n_requests):
        initial_pub_client.publish(REQUEST_QUEUE, payload)

    start_time = time.time()
    n_received = 0
    while True:
        pull_response = sub_client.pull(
            SUBSCRIPTION_NAME, SUBSCRIPTION_MAX_MESSAGES, return_immediately=True
        )
        end_time = time.time()

        ack_ids = []
        futures = []
        for message in pull_response.received_messages:
            futures.append(pub_client.publish(REQUEST_QUEUE, payload))
            ack_ids.append(message.ack_id)

        for f in futures:
            f.result()

        if ack_ids:
            sub_client.acknowledge(SUBSCRIPTION_NAME, ack_ids)

        n_received += len(ack_ids)
        dt = end_time - start_time
        if dt >= MIN_THROUGHPUT_TIME:
            print(f"Throughput: {n_received / dt}")
            start_time = end_time
            n_received = 0


if __name__ == "__main__":
    main()
