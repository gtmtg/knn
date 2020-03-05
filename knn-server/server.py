import functools
import heapq
from multiprocessing import dummy as multiprocessing

from flask import Flask, jsonify, render_template, request
import requests

import config


running = False
image_list = open(config.IMAGE_LIST_PATH, "r")
state_lock = multiprocessing.Lock()

results = []  # min-heap
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
        running_copy = running

        if running:
            image_list.seek(0)

    if running_copy:
        r = requests.post(
            f"{config.HANDLER_URL}{config.EMBEDDING_ENDPOINT}", json=request.get_json()
        )
        r.raise_for_status()
        worker_fn = functools.partial(thread_worker, template=r.text)

    with results_lock:
        results.clear()

        if running_copy:
            pool.imap_unordered(worker_fn, image_chunk_gen())
        else:
            pool.terminate()
            pool.join()

    return jsonify(running=running_copy)


@app.route("/results")
def get_results():
    with results_lock:
        sorted_results = sorted(results)

    def get_display_string(result_tuple):
        neg_score, image_name = result_tuple
        return f'{image_name} ({-neg_score})<br><img src="{config.IMAGE_URL_FORMAT.format(image_name)}" style="width: 250px;"/>'  # noqa

    return jsonify(results=list(map(get_display_string, sorted_results)))


def thread_worker(images, template):
    j = {"template": template, "images": images}
    r = requests.post(f"{config.HANDLER_URL}{config.INFERENCE_ENDPOINT}", json=j)
    chunk_results = r.json()

    with results_lock:
        for image_name, neg_score in chunk_results.items():
            result_tuple = (neg_score, image_name)
            if len(results) < config.N_RESULTS_TO_DISPLAY:
                heapq.heappush(results, result_tuple)
            elif result_tuple > results[0]:
                heapq.heapreplace(results, result_tuple)
