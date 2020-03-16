import collections
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
cost = 0

total_time = 0
total_gcr_time = 0
total_compute_time = 0

per_worker_num_processed = collections.defaultdict(int)

results = []  # min-heap
current_id = None
results_lock = multiprocessing.Lock()

pool = multiprocessing.Pool(processes=config.MAX_CONCURRENT_REQUESTS)


# Start web server
app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("index.html")


@app.route("/running")
def get_running():
    with state_lock:
        running_copy = running
    return jsonify(running=running_copy)


@app.route("/toggle", methods=["POST"])
def toggle():
    global running
    with state_lock:
        running_copy = running

    template = None
    if not running_copy:
        if request.get_json():
            for i in range(config.NUM_TEMPLATE_RETRIES):
                r = requests.post(
                    f"{config.HANDLER_URL}{config.EMBEDDING_ENDPOINT}",
                    json=request.get_json(),
                )
                if r.status_code == 200:
                    template = r.text
                    break
        if not template:
            return "Couldn't get template", 400

    with state_lock:
        running = not running
        if not running:
            return jsonify(running=False)

        image_list.seek(0)

    with results_lock:
        results.clear()
        per_worker_num_processed.clear()

        global num_processed
        global num_skipped
        global cost
        num_processed = 0
        num_skipped = 0
        cost = 0

        global total_time
        global total_gcr_time
        global total_compute_time
        total_time = 0
        total_gcr_time = 0
        total_compute_time = 0

        global current_id
        current_id = uuid.uuid4()

        pool.starmap_async(
            thread_worker, [(template, current_id)] * config.MAX_CONCURRENT_REQUESTS
        )

    return jsonify(running=True)


@app.route("/results")
def get_results():
    with state_lock:
        running_copy = running

    with results_lock:
        sorted_results = sorted(results)

        workers_copy = {
            i: n for i, (_, n) in enumerate(per_worker_num_processed.items())
        }

        cost_copy = cost
        num_processed_copy = num_processed
        num_skipped_copy = num_skipped

        if num_processed:
            avg_total_time = total_time / num_processed
            avg_gcr_time = total_gcr_time / num_processed
            avg_compute_time = total_compute_time / num_processed
        else:
            avg_total_time = 0
            avg_gcr_time = 0
            avg_compute_time = 0

    def get_display_string(result_tuple):
        score, image_name, score_map = result_tuple
        return f'{image_name} ({score})<br><img src="{config.IMAGE_URL_FORMAT.format(image_name)}" style="width: 250px;"/><img src="data:image/jpeg;base64,{score_map}" style="width: 250px;"/>'  # noqa

    return jsonify(
        stats=dict(
            cost=cost_copy,
            num_processed=num_processed_copy,
            num_skipped=num_skipped_copy,
            num_total=config.N_TOTAL_IMAGES,
            #
            total_time=avg_total_time,
            gcr_time=avg_gcr_time,
            compute_time=avg_compute_time,
        ),
        workers=workers_copy,
        results=[get_display_string(r) for r in reversed(sorted_results)],
        running=running_copy,
    )


def image_chunk_gen():
    global running
    while True:
        images = []
        with state_lock:
            if not running:
                break

            first_image = image_list.readline().rstrip()
            if not first_image:
                running = False
                break
            images.append(first_image)

            for _ in range(config.CHUNK_SIZE - 1):
                next_image = image_list.readline().rstrip()
                if next_image:
                    images.append(next_image)
        yield images


def thread_worker(template, id):
    for image_chunk in image_chunk_gen():
        j = {"template": template, "images": image_chunk, "include_score_map": True}
        try:
            r = requests.post(
                f"{config.HANDLER_URL}{config.INFERENCE_ENDPOINT}", json=j
            )
            r.raise_for_status()
            chunk_results = r.json()
        except Exception:
            chunk_results = {}

        with results_lock:
            if id != current_id:
                break

            metadata = chunk_results.pop("metadata", None)
            if metadata:
                global cost

                cost += (
                    0.00002400 * metadata["total"]
                    + 2 * 0.00000250 * metadata["total"]
                    + 0.40 / 1000000
                )

                global total_time
                global total_gcr_time
                global total_compute_time

                total_time += r.elapsed.total_seconds()
                total_gcr_time += metadata["total"]
                total_compute_time += metadata["compute"]

                per_worker_num_processed[metadata["worker_id"]] += 1

            global num_processed
            global num_skipped

            num_processed += len(chunk_results)
            num_skipped += len(image_chunk) - len(chunk_results)

            for image_name, result in chunk_results.items():
                result_tuple = (result["score"], image_name, result["score_map"])
                if len(results) < config.N_RESULTS_TO_DISPLAY:
                    heapq.heappush(results, result_tuple)
                elif result_tuple > results[0]:
                    heapq.heapreplace(results, result_tuple)

    return True
