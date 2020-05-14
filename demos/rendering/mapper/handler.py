import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from knn import utils
from knn.mappers import Mapper


MODEL = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"


class PanopticSegmentationMapper(Mapper):
    def initialize_container(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL))
        self.model = DefaultPredictor(cfg)

        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        torch.set_grad_enabled(False)

    async def process_element(
        self, input, job_id, job_args, request_id, element_index,
    ):
        # Deserialize
        frame = utils.base64_to_numpy(input["frame"])

        # RGB -> BGR
        frame_bgr = frame[:, :, ::-1]

        # Perform segmentation
        result = self.model(frame_bgr)

        # Draw output
        visualizer = Visualizer(frame, self.metadata)
        vis_output = visualizer.draw_panoptic_seg_predictions(*result["panoptic_seg"])

        # Serialize
        rendered = vis_output.get_image()
        return utils.numpy_to_base64(rendered)


mapper = PanopticSegmentationMapper()
