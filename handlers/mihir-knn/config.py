from detectron2.config.config import get_cfg as get_default_detectron_config

MODEL_CLOUD_PATH = "gs://mihir-knn/models/R-50.pkl"
MODEL_LOCAL_PATH = "/tmp/knn/R-50.pkl"

N_DISTANCES_TO_AVERAGE = 50
SERIALIZE_DTYPE = "float32"

MIN_VIZ_SCORE = 0.25
MAX_VIZ_SCORE = 0.75

RESNET_CONFIG = get_default_detectron_config()
