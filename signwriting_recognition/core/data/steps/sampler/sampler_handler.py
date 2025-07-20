from core.data.steps.sampler.sampler_function import create_sample
from core.dtype import AbstractHandler
from core.pipeline import DataPipeline
from dataclasses import dataclass
import logging
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
            logging.info(f"DataPipeline: Run Sampler - Factor = {self.factor}")
            request.last_intermediate_step_path = f"{request.target_path}/intermediate/1_sampler"
            
            if os.path.isdir(request.last_intermediate_step_path):
                try:
                    os.removedirs(request.last_intermediate_step_path)
                except:
                    pass
                
            create_sample(request.original_path, request.last_intermediate_step_path, self.factor)
            request.original_path = request.last_intermediate_step_path
        return super().handle(request)
