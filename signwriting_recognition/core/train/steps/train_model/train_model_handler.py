from core.train.steps.train_model.train_model_function import get_trainer_model
from core.dtype import AbstractHandler
from core.domain import TrainPipeline
from dataclasses import dataclass
import logging


@dataclass
class TrainModelHandler(AbstractHandler):

    def handle(self, request: TrainPipeline) -> TrainPipeline:
        for model in request.models:
            logging.info(f"TrainPipeline: Run Train Model {model.model_name}")
            trainer = get_trainer_model(model)
            history, model = trainer.train(request.train_dataset, request.val_dataset)
        return super().handle(request)
