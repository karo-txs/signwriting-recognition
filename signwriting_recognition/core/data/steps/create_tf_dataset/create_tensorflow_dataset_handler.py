from core.data.steps.create_tf_dataset.create_tensorflow_dataset_function import (
    create_dataset_from_dict,
)
from core.dtype import AbstractHandler
from core.pipeline import DataPipeline
from dataclasses import dataclass
import logging


@dataclass
class CreateTensorFlowDatasetHandler(AbstractHandler):

    def validate(self, request: DataPipeline) -> bool:
        for step in request.steps:
            if step.get("name") == "normalization":
                return True

        return False

    def handle(self, request: DataPipeline) -> DataPipeline:
        if self.validate(request):
            logging.info(f"DataPipeline: Run Create TensorFlow Dataset")
            request.last_intermediate_step_path = (
                f"{request.target_path}/intermediate/1_sampler"
            )
            request.last_intermediate_step_data = create_dataset_from_dict(
                request.last_intermediate_step_data.get("landmark_dict"),
                request.last_intermediate_step_data.get("hand_data_labels"),
            )
        return super().handle(request)
