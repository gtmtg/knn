from detectron2.config.config import get_cfg as get_default_detectron_config
import numpy as np

LOG_LEVEL = "INFO"

CLOUD_STORAGE_BUCKET = "mihir-knn"
MODEL_CLOUD_PATH = "models/R-50.pkl"
MODEL_LOCAL_PATH = "/tmp/knn/R-50.pkl"

RESNET_CONFIG = get_default_detectron_config()
RESNET_DOWNSAMPLE_FACTOR = 16

TEMPLATE_DTYPE = np.float32
