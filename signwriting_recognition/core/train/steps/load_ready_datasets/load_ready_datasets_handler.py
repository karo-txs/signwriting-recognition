from core.train.steps.load_ready_datasets.load_ready_datasets_function import load_ready_dataset
from core.utils.counter_functions import count_unique_classes
from core.dtype import AbstractHandler
from core.domain import TrainPipeline
from dataclasses import dataclass
import logging


@dataclass
class LoadReadyDatasetsHandler(AbstractHandler):

    def validate(self, request: TrainPipeline) -> bool:
        return True

    def handle(self, request: TrainPipeline) -> TrainPipeline:
        if self.validate(request):
            logging.info(f"TrainPipeline: Run LoadReadyDatasets")
            request.train_dataset = load_ready_dataset(f"{request.experiment_path}/datasets/train.tfrecord")
            request.val_dataset = load_ready_dataset(f"{request.experiment_path}/datasets/val.tfrecord")
            request.len_unique_classes, _ = count_unique_classes(request.train_dataset)

        return super().handle(request)
