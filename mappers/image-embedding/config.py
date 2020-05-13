from detectron2.config.config import get_cfg as get_default_detectron_config

WEIGHTS_BUCKET = "mihir-fast-queries-west"
WEIGHTS_CLOUD_PATH = "models/R-50.pkl"
WEIGHTS_LOCAL_PATH = "tmp/R-50.pkl"
RESNET_CONFIG = get_default_detectron_config()
