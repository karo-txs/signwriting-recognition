from core.eval.steps.load_model.load_model_function import load_models_from_experiment
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class LoadModelHandler(AbstractHandler):

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        logging.info(f"EvaluationPipeline: Run Load Model")
        request.actual_model = load_models_from_experiment(request.actual_model_path.get("path"))

        return super().handle(request)
