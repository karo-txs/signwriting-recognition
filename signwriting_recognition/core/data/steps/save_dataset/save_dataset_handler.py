import os
from core.data.steps.save_dataset.save_dataset_function import export_chunks
from dataclasses import dataclass, field
from core.pipeline import DataPipeline
from core.dtype import AbstractHandler
import logging


@dataclass
class SaveDatasetHandler(AbstractHandler):

    chunk_size: int = field(default=5000, init=False)

    def handle(self, request: DataPipeline) -> DataPipeline:
        logging.info(
            f"DataPipeline: SaveDataset (batch={self.chunk_size}) → "
            f"{request.target_path}/{request.dataset_name}"
        )

        export_chunks(
            tf_dataset=request.last_intermediate_step_data,
            chunk_size=self.chunk_size,
            exporter_dataset_path=request.target_path,
            dataset_name=request.dataset_name,
            executor_type="process",
            max_workers=os.cpu_count(),
        )

        return super().handle(request)
