from core.data.steps.normalization.normalization_function import (
    dataset_landmark_normalization,
)
from core.dtype import AbstractHandler
from core.domain import DataPipeline
from dataclasses import dataclass
import logging


@dataclass
class NormalizationHandler(AbstractHandler):

    def validate(self, request: DataPipeline) -> bool:
        for step in request.steps:
            if step.get("name") == "normalization":
                return True

        return False

    def handle(self, request: DataPipeline) -> DataPipeline:
        if self.validate(request):
            logging.info(f"DataPipeline: Run Normalization")
            request.last_intermediate_step_path = (
                f"{request.target_path}/intermediate/3_normalization"
            )
            request.last_intermediate_step_data = dataset_landmark_normalization(
                tf_dataset=request.last_intermediate_step_data,
                save_path=request.last_intermediate_step_path,
                save_landmark_image=request.save_intermediate_steps,
            )
        return super().handle(request)
