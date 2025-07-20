from core.data.steps.sampler.sampler_function import create_sample
from core.dtype import AbstractHandler
from core.pipeline import DataPipeline
from dataclasses import dataclass
import logging
import shutil
import os


@dataclass
class SamplerHandler(AbstractHandler):

    def validate(self, request: DataPipeline) -> bool:
        for step in request.steps:
            if step.get("name") == "sampler":
                self.factor = step.get("factor", 25)
                return True

        return False

    def handle(self, request: DataPipeline) -> DataPipeline:
        if self.validate(request):
            logging.info("DataPipeline: Run Sampler ‒ Factor = %s", self.factor)

            request.last_intermediate_step_path = (
                f"{request.target_path}/intermediate/1_sampler"
            )

            if os.path.isdir(request.last_intermediate_step_path):
                shutil.rmtree(request.last_intermediate_step_path)

            create_sample(
                source_path=request.original_path,
                target_path=request.last_intermediate_step_path,
                sample_size=self.factor,
            )

            request.original_path = request.last_intermediate_step_path

        return super().handle(request)
