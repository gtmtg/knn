import base64
import collections
import functools
import io
import math
import os
import pathlib
import time
import uuid

from flask import Flask, jsonify, request
from google.cloud import storage, pubsub
import numpy as np
from PIL import Image
import torch

import config


# Set up Google Cloud clients
storage_client = storage.Client()
publish_client = pubsub.PublisherClient()
storage_bucket = storage_client.bucket(config.CLOUD_STORAGE_BUCKET)


# Lazy load model
model = None


def ensure_globals(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs["start_time"] = time.time()

        global model
        if not model:
            from detectron2.layers import ShapeSpec
            from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
            from detectron2.modeling.backbone.resnet import build_resnet_backbone

            # Download model weights
            pathlib.Path(config.MODEL_LOCAL_PATH).parent.mkdir(
                parents=True, exist_ok=True
            )
            model_blob = storage_bucket.blob(config.MODEL_CLOUD_PATH)
            model_blob.download_to_filename(config.MODEL_LOCAL_PATH)

            # Create model
            shape = ShapeSpec(channels=3)
            model = torch.nn.Sequential(
                build_resnet_backbone(config.get_resnet_config(), shape)
            )

            # Load model weights
            checkpointer = DetectionCheckpointer(model, save_to_disk=False)
            checkpointer.load(config.MODEL_LOCAL_PATH)
            model.eval()

            # Delete weights file to free up memory
            os.remove(config.MODEL_LOCAL_PATH)

        return f(*args, **kwargs)

    return wrapper


# Start web server
worker_id = uuid.uuid4()
app = Flask(__name__)


@app.route("/", methods=["POST"])
@ensure_globals
def handler(start_time):
    request_start_time = time.time()

    event = request.get_json()
    template = deserialize(event["template"]).unsqueeze(dim=0)

    results = {
        "worker_id": worker_id,
        "compute_time": 0.0,
        "images": collections.defaultdict(dict),
    }

    for image_cloud_path in event["images"]:
        try:
            image = download_image(image_cloud_path)

            compute_start_time = time.time()
            result = run_inference(image)
            score_map, score = compute_knn_score(result, template)
            results["compute_time"] += time.time() - compute_start_time

            results["images"][image_cloud_path]["score"] = score
            if event.get("include_score_map"):
                score_map_base64 = float_map_to_base64_png(score_map)
                results["images"][image_cloud_path]["score_map"] = score_map_base64
        except AssertionError:
            pass

    results["request_time"] = time.time() - request_start_time
    results["gcr_time"] = time.time() - start_time

    return jsonify(results)


@app.route("/get_embedding", methods=["POST"])
@ensure_globals
def get_embedding(start_time):
    event = request.get_json()

    image = download_image(event["image"])
    result = run_inference(image)

    x1, y1, x2, y2 = event["patch"]
    x1, y1 = [math.floor(dim / config.RESNET_DOWNSAMPLE_FACTOR) for dim in (x1, y1)]
    x2, y2 = [math.ceil(dim / config.RESNET_DOWNSAMPLE_FACTOR) for dim in (x2, y2)]

    embedding = result[:, y1:y2, x1:x2].mean(dim=-1).mean(dim=-1)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return serialize(embedding)


def download_image(image_cloud_path):
    image_buffer = io.BytesIO()
    image_blob = storage_bucket.blob(image_cloud_path)
    image_blob.download_to_file(image_buffer)
    image = Image.open(image_buffer)
    image = image_to_tensor(image)
    image_buffer.close()
    return image


def run_inference(image):
    with torch.no_grad():
        result = model(image)
    assert len(result) == 1
    return next(iter(result.values())).squeeze(dim=0)


def image_to_tensor(image):
    image = np.asarray(image, dtype=np.float32)
    assert image.ndim == 3

    with torch.no_grad():
        # HWC -> NCHW
        image = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(dim=0)

        # RGB -> BGR
        if config.get_resnet_config().INPUT.FORMAT == "BGR":
            image = torch.flip(image, dims=(1,))
        image = image.contiguous()

        # Normalize
        pixel_mean = torch.Tensor(config.get_resnet_config().MODEL.PIXEL_MEAN).view(
            1, -1, 1, 1
        )
        pixel_std = torch.Tensor(config.get_resnet_config().MODEL.PIXEL_STD).view(
            1, -1, 1, 1
        )
        return (image - pixel_mean) / pixel_std


def compute_knn_score(embeddings, template):
    def cosine_similarity(x1_n, x2_t_n):  # n = L2 normalized, t = transposed
        return torch.mm(x1_n, x2_t_n)

    with torch.no_grad():
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        embeddings_flat = torch.nn.functional.normalize(embeddings_flat, p=2, dim=0)
        scores = cosine_similarity(template, embeddings_flat).view(-1)
        score_map = scores.view(embeddings.size(1), embeddings.size(2))
        score = torch.topk(scores, config.N_DISTANCES_TO_AVERAGE).values.mean().item()
        return score_map, score


def serialize(tensor):
    flat = tensor.view(-1)
    return base64.b64encode(flat.numpy().astype(config.SERIALIZE_DTYPE))


def float_map_to_base64_png(float_map):
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    with torch.no_grad():
        float_map = 255 * rescale(float_map)

    image = Image.fromarray(float_map.numpy().astype(np.uint8))
    image_buffer = io.BytesIO()
    image.save(image_buffer, "png")
    image_buffer.seek(0)
    return base64.b64encode(image_buffer.read()).decode("ascii")


def deserialize(base64_string):
    decoded = base64.b64decode(base64_string)
    numpy = np.frombuffer(decoded, dtype=config.SERIALIZE_DTYPE)
    tensor = torch.as_tensor(numpy)
    return tensor
