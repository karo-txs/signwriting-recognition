from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
import os
from core.data.steps.chunker.chunker_function import (
    load_chunk_data,
    save_chunk_dataset,
)
from core.dtype import AbstractHandler
from core.domain import DataPipeline
from dataclasses import dataclass
import logging


@dataclass
class ChunkerHandler(AbstractHandler):

    data_process: AbstractHandler
    CHUNK_SIZE = 500

    def validate(self, request: DataPipeline) -> bool:
        for step in request.steps:
            if step.get("name") == "landmark-detector":
                return True
        return False

    def _process_single_chunk(self, base_request: DataPipeline, chunk_data, idx: int):
        req = copy.copy(base_request)
        req.last_intermediate_step_data = chunk_data

        req = self.data_process.handle(req)

        save_chunk_dataset(
            req.last_intermediate_step_data,
            exporter_dataset_path=req.target_path,
            dataset_name=req.dataset_name,
            chunk_name=idx,
        )
        return f"Chunk {idx} salvo."

    def handle(self, request: DataPipeline) -> DataPipeline:
        if not self.validate(request):
            return super().handle(request)

        logging.info(f"DataPipeline: Run ImageChunker - chunk_size = {self.CHUNK_SIZE}")

        origin_chunk_data = request.last_intermediate_step_path or request.original_path
        generator = load_chunk_data(origin_chunk_data, chunk_size=self.CHUNK_SIZE)

        base_req = copy.copy(request)
        base_req.last_intermediate_step_data = None
        base_req.last_intermediate_step_path = None

        max_workers = min(32, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for idx, chunk in enumerate(generator):
                futures.append(
                    pool.submit(self._process_single_chunk, base_req, chunk, idx)
                )

            for fut in as_completed(futures):
                logging.info(fut.result())

        return super().handle(request)
