from core.train.steps.prepare_datasets.prepare_datasets_function_v2 import prepare_dataset
from core.pipeline import TrainPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging
import os


@dataclass
class PrepareDatasetsHandler(AbstractHandler):

    def validate(self, request: TrainPipeline) -> bool:
        return (
            request.train_dataset_paths is not None
            and request.val_dataset_paths is not None
            and request.test_dataset_paths is not None
            and not os.path.isdir(f"{request.experiment_path}/datasets/split/train/")
        )

    def handle(self, request: TrainPipeline) -> TrainPipeline:
        if self.validate(request):
            logging.info(f"TrainPipeline: Run PrepareDatasets - Train")
            prepare_dataset(
                datasets_path=request.train_dataset_paths,
                split="train",
                experiment_path=f"{request.experiment_path}/datasets/",
                label_names=request.label_names,
            )

            logging.info(f"TrainPipeline: Run PrepareDatasets - Val")
            prepare_dataset(
                datasets_path=request.val_dataset_paths,
                split="val",
                experiment_path=f"{request.experiment_path}/datasets/",
                label_names=request.label_names,
            )

            logging.info(f"TrainPipeline: Run PrepareDatasets - Test")
            prepare_dataset(
                datasets_path=request.test_dataset_paths,
                split="test",
                experiment_path=f"{request.experiment_path}/datasets/",
                label_names=request.label_names,
            )

        return super().handle(request)
