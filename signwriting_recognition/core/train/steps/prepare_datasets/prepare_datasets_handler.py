from core.train.steps.prepare_datasets.prepare_datasets_function import prepare_dataset
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
            folders = [d["path"] for d in request.train_dataset_paths]
            prepare_dataset(
                folders=folders,
                split="train",
                experiment_path=f"{request.experiment_path}/datasets/",
                label_names=request.label_names,
            )

            logging.info(f"TrainPipeline: Run PrepareDatasets - Val")
            folders = [d["path"] for d in request.val_dataset_paths]
            prepare_dataset(
                folders=folders,
                split="val",
                experiment_path=f"{request.experiment_path}/datasets/",
                label_names=request.label_names,
            )

        return super().handle(request)
