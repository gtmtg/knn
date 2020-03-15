import functools
import heapq
from multiprocessing import dummy as multiprocessing
import uuid

from flask import Flask, jsonify, render_template, request
import requests

import config


running = False
image_list = open(config.IMAGE_LIST_PATH, "r")
state_lock = multiprocessing.Lock()

num_processed = 0
num_skipped = 0
results = []  # min-heap
current_id = None
results_lock = multiprocessing.Lock()

pool = multiprocessing.Pool(processes=config.MAX_CONCURRENT_REQUESTS)


# Start web server
app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("index.html")


def image_chunk_gen():
    while True:
        images = []
        with state_lock:
            if not running:
                return

            first_image = image_list.readline().rstrip()
            if not first_image:
                return
            images.append(first_image)

            for _ in range(config.CHUNK_SIZE - 1):
                next_image = image_list.readline().rstrip()
                if next_image:
                    images.append(next_image)
        yield images


@app.route("/running")
def get_running():
    with state_lock:
        running_copy = running
    return jsonify(running=running_copy)


@app.route("/toggle", methods=["POST"])
def toggle():
    with state_lock:
        global running
        running = not running
        if not running:
            return jsonify(running=False)

        image_list.seek(0)

    r = requests.post(
        f"{config.HANDLER_URL}{config.EMBEDDING_ENDPOINT}", json=request.get_json()
    )
    r.raise_for_status()

    with results_lock:
        global num_processed
        global num_skipped
        global current_id
        num_processed = 0
        num_skipped = 0
        results.clear()
        current_id = uuid.uuid4()

        worker_fn = functools.partial(thread_worker, template=r.text, id=current_id)
        pool.imap_unordered(worker_fn, image_chunk_gen())

    return jsonify(running=True)


@app.route("/results")
def get_results():
    with results_lock:
        sorted_results = sorted(results)
        num_processed_copy = num_processed
        num_skipped_copy = num_skipped

    def get_display_string(result_tuple):
        score, image_name, score_map = result_tuple
        return f'{image_name} ({score})<br><img src="{config.IMAGE_URL_FORMAT.format(image_name)}" style="width: 250px;"/><img src="data:image/jpeg;base64,{score_map}" style="width: 250px;"/>'  # noqa

    return jsonify(
        num_processed=num_processed_copy,
        num_skipped=num_skipped_copy,
        results=[get_display_string(r) for r in reversed(sorted_results)],
    )


def thread_worker(images, template, id):
    j = {"template": template, "images": images}
    try:
        r = requests.post(f"{config.HANDLER_URL}{config.INFERENCE_ENDPOINT}", json=j)
        r.raise_for_status()
        chunk_results = r.json()
    except Exception:
        chunk_results = {}

    with results_lock:
        if id == current_id:
            global num_processed
            global num_skipped
            num_processed += len(chunk_results)
            num_skipped += len(images) - len(chunk_results)
            for image_name, result in chunk_results.items():
                result_tuple = (result["score"], image_name, result["score_map"])
                if len(results) < config.N_RESULTS_TO_DISPLAY:
                    heapq.heappush(results, result_tuple)
                elif result_tuple > results[0]:
                    heapq.heapreplace(results, result_tuple)
        else:
            print(id, current_id)

    return True
