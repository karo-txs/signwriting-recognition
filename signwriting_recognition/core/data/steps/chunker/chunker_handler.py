from core.data.steps.chunker.chunker_function import (
    load_chunk_data,
    save_chunk_dataset,
)
from core.utils.tf_data_functions import (
    create_concatenated_dataset_from_folder,
    read_map_fn_with_str_label,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.utils.counter_functions import count_sample_per_class
from core.dtype import AbstractHandler
from core.pipeline import DataPipeline
from dataclasses import dataclass
import logging
import json
import copy
import os


@dataclass
class ChunkerHandler(AbstractHandler):

    data_process: AbstractHandler
    CHUNK_SIZE = 3000

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

        self.save_data_info(
            dataset_path=f"{base_req.target_path}/{base_req.dataset_name}",
            dtype=str,
        )

        return super().handle(request)

    def save_data_info(self, dataset_path, dtype=str):
        json_path = os.path.join(dataset_path, "info.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data_info = json.load(f)
        else:
            data_info = {"total_samples": 0, "classes": {}}

        dataset = create_concatenated_dataset_from_folder(
            folder_path=dataset_path, read_map_fn=read_map_fn_with_str_label
        )
        count_classes = count_sample_per_class(dataset, dtype=dtype)

        for cls, n in count_classes.items():
            data_info["classes"][cls] = data_info["classes"].get(cls, 0) + n
        data_info["total_samples"] = sum(data_info["classes"].values())

        with open(json_path, "w") as f:
            json.dump(data_info, f, indent=4)
        return json_path
