from core.eval.steps.classification_report.classification_function import (
    load_class_names,
    save_full_report,
)
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
import logging


@dataclass
class ClassificationReportHandler(AbstractHandler):

    def validate(self, request: EvaluationPipeline) -> bool:
        return True

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        if self.validate(request):
            logging.info(f"EvaluationPipeline: Run Classification Report")
            class_names = load_class_names(request.experiment_path)

            for model in request.models:
                logging.info(
                    f"EvaluationPipeline: {model.name} ({model.framework.name})"
                )
                model.metrics = model.predict_dataset(request.test_dataset)
                save_full_report(
                    model_path=f"{request.experiment_path}/models/{model.name}",
                    metrics_dict=model.metrics,
                    class_names=class_names,
                )

        return super().handle(request)
