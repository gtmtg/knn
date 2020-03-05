# import http
# import json

from io import BytesIO
import math
import os
from pathlib import Path

from flask import Flask, jsonify, request
from google.cloud import storage, pubsub
import numpy as np
from PIL import Image
import torch

from detectron2.layers import ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone.resnet import build_resnet_backbone

import config


# Set up Google Cloud clients
storage_client = storage.Client()
publish_client = pubsub.PublisherClient()

# Download model weights
storage_bucket = storage_client.bucket(config.CLOUD_STORAGE_BUCKET)
Path(config.MODEL_LOCAL_PATH).parent.mkdir(parents=True, exist_ok=True)
model_blob = storage_bucket.blob(config.MODEL_CLOUD_PATH)
model_blob.download_to_filename(config.MODEL_LOCAL_PATH)

# Create model
shape = ShapeSpec(channels=3)
model = torch.nn.Sequential(build_resnet_backbone(config.RESNET_CONFIG, shape))

# Load model weights
checkpointer = DetectionCheckpointer(model, save_to_disk=False)
weights = checkpointer._load_file(config.MODEL_LOCAL_PATH)
checkpointer._load_model(weights)
model.eval()

# Delete weights file to free up memory
os.remove(config.MODEL_LOCAL_PATH)

# Start web server
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handler():
    event = request.get_json()
    results = {}

    template = np.fromstring(event["template"], dtype=config.TEMPLATE_DTYPE)
    template = torch.from_numpy(template).unsqueeze(dim=0).unsqueeze(dim=0)

    for image_cloud_path in event["images"]:
        try:
            result = run_inference(image_cloud_path)
            results[image_cloud_path] = compute_knn_score(result, template)
        except AssertionError:
            pass

    return jsonify(results)

    # publish_client.publish(event["id"], json.dumps(results))
    # return "", http.client.NO_CONTENT


@app.route("/get_embedding", method=["POST"])
def get_embedding():
    event = request.get_json()

    result = run_inference(event["image"])

    x1, y1, x2, y2 = event["patch"]
    x1, y1 = [math.floor(dim / config.RESNET_DOWNSAMPLE_FACTOR) for dim in (x1, y1)]
    x2, y2 = [math.ceil(dim / config.RESNET_DOWNSAMPLE_FACTOR) for dim in (x2, y2)]

    embedding = result[0, :, y1:y2, x1:x2].mean(dim=-1).mean(dim=-1)
    return embedding.astype(config.TEMPLATE_DTYPE).tostring()


def run_inference(image_cloud_path):
    # Load image
    image_buffer = BytesIO()
    image_blob = storage_bucket.blob(image_cloud_path)
    image_blob.download_to_file(image_buffer)
    image = Image.open(image_buffer)
    image = np.asarray(image, dtype=np.float32)
    assert image.ndim == 3
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)  # NCHW

    # Run inference
    with torch.no_grad():
        result = model(image)
    assert len(result) == 1
    return next(iter(result.values()))


def compute_knn_score(image, template):
    assert image.size[0] == 1  # batch dimension
    image = image.view(1, image.size[1], -1).permute(0, 2, 1)  # N(HW)C

    distances = torch.cdist(image, template)
    return torch.topk(distances, config.N_DISTANCES_TO_AVERAGE).values.mean()
