from lvis_config import IMAGE_BUCKET, IMAGE_LIST_PATH, DEMO_IMAGES

N_CONCURRENT_WORKERS_DEFAULT = 500

TEMPLATE_ENDPOINT = "https://mihir-image-embedding-g6rwrca4fq-uc.a.run.app"

QUERY_ENDPOINT = "https://mihir-spatial-search-g6rwrca4fq-uc.a.run.app"
OUTPUT_BUCKET = IMAGE_BUCKET
OUTPUT_PATH = "results/"
N_DISTANCES_TO_AVERAGE = 50

N_RESULTS_TO_DISPLAY = 50

QUERY_CLEANUP_TIME = 60 * 60  # seconds
