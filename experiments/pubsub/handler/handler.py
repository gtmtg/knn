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
    if "delay" in request.get_json():  # normal request
        delay = float(request.get_json()["delay"])
        time.sleep(delay)
        return request.get_json()
    else:  # PubSub
        message = base64.b64decode(request.get_json()["message"]["data"])
        delay = struct.unpack("f", message)[0]

        time.sleep(delay)

        future = publish_client.publish(RESPONSE_QUEUE, message)
        future.result()  # block until message is actually published
        return "", 204
