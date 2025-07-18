from core.eval.steps.load_ready_datasets.load_ready_datasets_function import load_ready_dataset
from core.utils.counter_functions import count_unique_classes
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class LoadReadyDatasetsHandler(AbstractHandler):

    def validate(self, request: EvaluationPipeline) -> bool:
        return True

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        if self.validate(request):
            logging.info(f"EvaluationPipeline: Run Load Ready Datasets")
            request.test_dataset = load_ready_dataset(f"{request.experiment_path}/datasets/split/test")
            request.len_unique_classes, _ = count_unique_classes(request.test_dataset)

        return super().handle(request)
