from core.data.steps.augmentation.augmentation_function import landmark_augmentation
from core.dtype import AbstractHandler
from core.pipeline import DataPipeline
from dataclasses import dataclass
import logging


@dataclass
class AugmentationHandler(AbstractHandler):

    def validate(self, request: DataPipeline) -> bool:
        for step in request.steps:
            if step.get("name") == "augmentation":
                self.factor = step.get("factor", 5)
                self.methods = step.get("methods", ["rotate_finger"])
                return True

        return False

    def handle(self, request: DataPipeline) -> DataPipeline:
        if self.validate(request):
            logging.info(f"DataPipeline: Run Augmentation")
            request.last_intermediate_step_path = (
                f"{request.target_path}/intermediate/4_augmentation"
            )

            request.last_intermediate_step_data = landmark_augmentation(
                tf_dataset=request.last_intermediate_step_data,
                generate_methods=self.methods,
                max_gestures=self.factor,
                save_path=request.last_intermediate_step_path,
                save_landmark_image=request.save_intermediate_steps,
            )
        return super().handle(request)
