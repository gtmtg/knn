import base64
import struct
import time

from flask import Flask, request
from google.cloud import pubsub


RESPONSE_QUEUE = "projects/{project_id}/topics/{topic}".format(
    project_id="mihir-knn", topic="results"
)


publish_client = pubsub.PublisherClient(
    pubsub.types.BatchSettings(max_messages=1)  # no batching
)


app = Flask(__name__)


@app.route("/", methods=["POST"])
def handler():
    message = base64.b64decode(request.get_json()["message"]["data"])
    delay = struct.unpack("f", message)

    # Simulate computation
    time.sleep(delay)

    publish_client.publish(RESPONSE_QUEUE, message)
    return "", 204
