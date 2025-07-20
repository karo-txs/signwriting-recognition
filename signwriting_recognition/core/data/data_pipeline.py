from core.pipeline import DataPipeline
from core.data.steps import (
    LandmarkDetectionHandler,
    CreateTensorFlowDatasetHandler,
    NormalizationHandler,
    SamplerHandler,
    SaveDatasetHandler,
    AugmentationHandler,
)
import logging


def run_data_pipeline(data_pipeline_config: DataPipeline) -> DataPipeline:

    logging.info(f"Processing: {data_pipeline_config.original_path}")
    base_pipeline = SamplerHandler()
    base_pipeline.set_next(LandmarkDetectionHandler()).set_next(CreateTensorFlowDatasetHandler()).set_next(
        NormalizationHandler()
    ).set_next(AugmentationHandler()).set_next(SaveDatasetHandler())

    result = base_pipeline.handle(data_pipeline_config)

    return result
