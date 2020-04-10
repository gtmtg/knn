from lvis_config import IMAGE_LIST_PATH, DEMO_IMAGES

CHUNK_SIZE = 3
N_RESULTS_TO_DISPLAY = 50
N_CONCURRENT_WORKERS_RANGE = (1, 350, 1000)  # min, default, max

HANDLER_URL = "https://knn-wj4n6yaj3q-uw.a.run.app"
INFERENCE_ENDPOINT = "/"
TEMPLATE_ENDPOINT = "/get_embedding"
IMAGE_URL_FORMAT = "https://storage.googleapis.com/mihir-knn/{}"

N_TEMPLATE_ATTEMPTS = 5
N_QUERY_ATTEMPTS = 3
