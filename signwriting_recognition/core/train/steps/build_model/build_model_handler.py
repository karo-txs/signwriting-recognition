from core.train.steps.build_model.build_model_function import build_model
from core.dtype import AbstractHandler
from core.domain import TrainPipeline
from dataclasses import dataclass
import logging


@dataclass
class BuildModelHandler(AbstractHandler):

    def handle(self, request: TrainPipeline) -> TrainPipeline:
        models = []
        for model_params in request.models_params:
            model_name = model_params.get("model")
            logging.info(f"TrainPipeline: Run Build Model {model_name}")
            models.append(
                build_model(
                    model_params,
                    request.len_unique_classes,
                    f"{request.experiment_path}/models/{model_name}",
                )
            )
        request.models = models
        return super().handle(request)
