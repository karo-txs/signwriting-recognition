from core.domain import DataPipeline
from core.data.steps import (
    LandmarkDetectionHandler,
    CreateTensorFlowDatasetHandler,
    NormalizationHandler,
    SamplerHandler,
    ChunkerHandler,
    AugmentationHandler,
)
import logging


def run_signwriting_data_pipeline(data_pipeline_config: DataPipeline) -> DataPipeline:

    logging.info(f"Processing: {data_pipeline_config.original_path}")
    base_pipeline = LandmarkDetectionHandler()
    base_pipeline.set_next(CreateTensorFlowDatasetHandler()).set_next(
        NormalizationHandler()
    ).set_next(AugmentationHandler())

    initial_step = SamplerHandler()
    initial_step.set_next(ChunkerHandler(base_pipeline))

    result = initial_step.handle(data_pipeline_config)

    return result
