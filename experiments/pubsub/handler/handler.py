import struct
import time

from flask import Flask, request
from google.cloud import pubsub


publish_client = pubsub.PublisherClient(
    pubsub.types.BatchSettings(max_messages=1)  # no batching
)


app = Flask(__name__)


@app.route("/", methods=["POST"])
def handler():
    delay = request.get_json()["delay"]
    response = struct.pack("f", delay)

    # Simulate computation
    time.sleep(delay)

    publish_client.publish("results", response)
    return "", 200
