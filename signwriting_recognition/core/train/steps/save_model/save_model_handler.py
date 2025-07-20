from core.train.steps.save_model.save_model_function import save_model
from core.pipeline import TrainPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class SaveModelHandler(AbstractHandler):

    def handle(self, request: TrainPipeline) -> TrainPipeline:
        for model in request.models:
            logging.info(f"TrainPipeline: Run Save Model {model.model_name}")
            save_model(model)
        return super().handle(request)
