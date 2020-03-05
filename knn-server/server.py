from functools import reduce
import heapq
from multiprocessing import dummy as multiprocessing
from operator import mul

from flask import Flask, jsonify, render_template
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


@app.route("/turn_on", methods=["POST"])
def turn_on():
    should_be_running = True
    with state_lock:
        global running
        running = should_be_running
        if running:
            image_list.seek(0)

    if should_be_running:
        with results_lock:
            results.clear()
        # Start pool
        pool.map(thread_worker, [0] * config.MAX_CONCURRENT_REQUESTS)
    else:
        # Kill pool
        with results_lock:
            results.clear()
        pool.terminate()
        pool.join()
    return ""


@app.route("/turn_off", methods=["POST"])
def turn_off():
    should_be_running = False
    with state_lock:
        global running
        running = should_be_running
        if running:
            image_list.seek(0)

    if should_be_running:
        with results_lock:
            results.clear()
        # Start pool
        pool.map(thread_worker, [0] * config.MAX_CONCURRENT_REQUESTS)
    else:
        # Kill pool
        with results_lock:
            results.clear()
        pool.terminate()
        pool.join()
    return ""


@app.route("/get_results")
def get_results():
    with results_lock:
        sorted_results = sorted(results)

    def get_display_string(result_tuple):
        _, shape, image_name = result_tuple
        dimensions = " &times; ".join(map(str, shape))
        return f"{image_name} ({dimensions})"

    response = jsonify(
        results=[get_display_string(r) for r in reversed(sorted_results)]
    )
    print(response)
    response.status_code = 200
    return response


@app.route("/")
def homepage():
    return render_template("index.html")


def thread_worker(_):
    while True:
        with state_lock:
            if not running:
                return
            images = [image_list.readline().rstrip() for _ in range(config.CHUNK_SIZE)]

        chunk_results = requests.post(
            config.INFERENCE_URL, json={"images": images}
        ).json()["results"]

        with results_lock:
            for image_name, shape in chunk_results.items():
                result_tuple = (reduce(mul, shape[2:], 1), shape, image_name)
                if len(results) < config.N_RESULTS_TO_DISPLAY:
                    heapq.heappush(results, result_tuple)
                elif result_tuple > results[0]:
                    heapq.heapreplace(results, result_tuple)
