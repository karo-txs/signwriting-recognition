from core.eval.steps import (
    LoadModelHandler,
    LoadReadyDatasetsHandler,
    PrepareDatasetsHandler,
    ClassificationReportHandler,
    BootstrapReportHandler,
)
from core.pipeline import EvaluationPipeline
import logging


def run_eval_pipeline(eval_pipeline_config: EvaluationPipeline) -> EvaluationPipeline:

    for test_dataset in eval_pipeline_config.test_dataset_paths:
        for model in eval_pipeline_config.models_path:
            logging.info(f"EvaluationPipeline: {test_dataset} - {model}")
            eval_pipeline_config.actual_test_dataset_path = test_dataset
            eval_pipeline_config.actual_model_path = model

            initial_step = PrepareDatasetsHandler()
            initial_step.set_next(LoadReadyDatasetsHandler()).set_next(
                LoadModelHandler()
            ).set_next(ClassificationReportHandler()).set_next(BootstrapReportHandler())

            result = initial_step.handle(eval_pipeline_config)

    return result
