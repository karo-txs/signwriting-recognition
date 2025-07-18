from core.eval.steps.load_model.load_model_function import load_models_from_experiment
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class LoadModelHandler(AbstractHandler):

    def validate(self, request: EvaluationPipeline) -> bool:
        return True

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        if self.validate(request):
            logging.info(f"EvaluationPipeline: Run Load Model")
            request.models = load_models_from_experiment(request.experiment_path)

        return super().handle(request)
