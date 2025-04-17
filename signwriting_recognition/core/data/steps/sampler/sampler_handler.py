from core.data.steps.sampler.sampler_function import create_sample
from core.dtype import AbstractHandler
from core.domain import DataPipeline
from dataclasses import dataclass
import logging


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
            logging.info(f"DataPipeline: Run Sampler - Factor = {self.factor}")
            request.last_intermediate_step_path = f"{request.target_path}/intermediate/1_sampler"
            create_sample(request.original_path, request.last_intermediate_step_path, self.factor)
        return super().handle(request)
