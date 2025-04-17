from core.data.steps.landmark_detector.mediapipe_landmark_detector import Mediapipe
from core.utils.performance_functions import measure_time
from core.utils.dict_functions import add_to_dict
from core.dtype import AbstractHandler
from core.domain import DataPipeline
from dataclasses import dataclass
import logging


@dataclass
class LandmarkDetectionHandler(AbstractHandler):

    def validate(self, request: DataPipeline) -> bool:
        for step in request.steps:
            if step.get("name") == "landmark-detector":
                self.model = step.get("model", "mediapipe")
                if self.model == "mediapipe":
                    self.hand_landmark_detector = Mediapipe()
                else:
                    return False
                return True
        return False

    @measure_time
    def handle(self, request: DataPipeline) -> DataPipeline:
        if self.validate(request):
            logging.info(
                f"DataPipeline: Run Hand Landmark Detection - model = {self.model}"
            )
            hand_data_labels = []
            landmark_dict = {}

            request.last_intermediate_step_path = (
                f"{request.target_path}/intermediate/2_landmark_detector"
            )

            for image_path in request.last_intermediate_step_data.keys():
                landmarks = self.hand_landmark_detector.detect_from_chunks(
                    image_path=image_path,
                    label=request.last_intermediate_step_data[image_path],
                    save_path=request.last_intermediate_step_path,
                    save_landmark_image=request.save_intermediate_steps,
                )

                if landmarks:
                    hand_data_labels.append(
                        request.last_intermediate_step_data[image_path]
                    )
                    add_to_dict(landmark_dict, landmarks)

            request.last_intermediate_step_data = {
                "hand_data_labels": hand_data_labels,
                "landmark_dict": landmark_dict,
            }

        return super().handle(request)
