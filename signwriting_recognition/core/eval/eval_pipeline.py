from core.eval.steps import (
    LoadModelHandler,
    LoadReadyDatasetsHandler,
    ClassificationReportHandler,
    BootstrapReportHandler,
)
from core.pipeline import EvaluationPipeline


def run_eval_pipeline(eval_pipeline_config: EvaluationPipeline) -> EvaluationPipeline:

    initial_step = LoadReadyDatasetsHandler()
    initial_step.set_next(LoadModelHandler()).set_next(
        ClassificationReportHandler()
    ).set_next(BootstrapReportHandler())

    result = initial_step.handle(eval_pipeline_config)

    return result
