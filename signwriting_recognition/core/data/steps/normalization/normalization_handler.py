from core.data.steps.normalization.normalization_function import (
    dataset_landmark_normalization,
)
from core.pipeline import DataPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class NormalizationHandler(AbstractHandler):

    def validate(self, request: DataPipeline) -> bool:
        step_cfg = next(
            (s for s in request.steps if s.get("name") == "normalization"), None
        )
        if not step_cfg:
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
