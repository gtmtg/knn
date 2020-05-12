from detectron2.config.config import get_cfg as get_default_detectron_config

RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.WEIGHTS = "R-50.pkl"  # model will be downloaded here during build

RESNET_DOWNSAMPLE_FACTOR = 16
