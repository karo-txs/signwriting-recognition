from core.train.steps.prepare_datasets.prepare_datasets_function import prepare_dataset
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging
import os


@dataclass
class PrepareDatasetsHandler(AbstractHandler):

    def validate(self, request: EvaluationPipeline) -> bool:
        name = request.actual_test_dataset_path.get("name")
        return not os.path.isdir(f"{request.eval_path}/{name}")

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        if self.validate(request):
            logging.info(f"TrainPipeline: Run PrepareDatasets - Test")
            name = request.actual_test_dataset_path.get("name")
            prepare_dataset(
                folders=[request.actual_test_dataset_path.get("path")],
                split=f"{name}/tfrecords",
                experiment_path=f"{request.eval_path}/",
                label_names=request.label_names,
            )

        return super().handle(request)
