CLOUD_STORAGE_BUCKET = "mihir-knn"
MODEL_CLOUD_PATH = "models/R-50.pkl"
MODEL_LOCAL_PATH = "/tmp/knn/R-50.pkl"

RESNET_DOWNSAMPLE_FACTOR = 16

N_DISTANCES_TO_AVERAGE = 50

SERIALIZE_DTYPE = "float32"


_RESNET_CONFIG = None


def get_resnet_config():
    global _RESNET_CONFIG
    if not _RESNET_CONFIG:
        from detectron2.config.config import get_cfg as get_default_detectron_config

        _RESNET_CONFIG = get_default_detectron_config()
    return _RESNET_CONFIG
