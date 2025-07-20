from core.train.steps import (
    PrepareDatasetsHandler,
    LoadReadyDatasetsHandler,
    BuildModelHandler,
    TrainModelHandler,
    SaveModelHandler,
)
from core.pipeline import TrainPipeline


def run_train_pipeline(train_pipeline_config: TrainPipeline) -> TrainPipeline:

    initial_step = PrepareDatasetsHandler()
    initial_step.set_next(LoadReadyDatasetsHandler()).set_next(
        BuildModelHandler()
    ).set_next(TrainModelHandler()).set_next(SaveModelHandler())

    result = initial_step.handle(train_pipeline_config)

    return result
