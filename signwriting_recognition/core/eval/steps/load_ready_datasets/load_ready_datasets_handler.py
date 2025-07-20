from core.eval.steps.load_ready_datasets.load_ready_datasets_function import (
    load_ready_dataset,
)
from core.utils.counter_functions import count_unique_classes
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class LoadReadyDatasetsHandler(AbstractHandler):

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        logging.info(f"EvaluationPipeline: Run Load Ready Datasets")
        name = request.actual_test_dataset_path.get("name")

        request.actual_test_dataset = load_ready_dataset(
            f"{request.eval_path}/{name}/tfrecords"
        )
        request.len_unique_classes, _ = count_unique_classes(
            request.actual_test_dataset
        )

        return super().handle(request)
