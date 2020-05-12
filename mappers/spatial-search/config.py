from detectron2.config.config import get_cfg as get_default_detectron_config

RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.WEIGHTS = "R-50.pkl"  # model will be downloaded here during build

MIN_VIZ_SCORE = 0.25
MAX_VIZ_SCORE = 0.75
