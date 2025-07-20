from core.eval.steps.classification_report.classification_function import (
    load_class_names,
    save_full_report,
)
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class ClassificationReportHandler(AbstractHandler):

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        logging.info(f"EvaluationPipeline: Run Classification Report")
        test_name = request.actual_test_dataset_path.get("name")
        dataset_dir = Path(request.eval_path) / test_name / "tfrecords"
        class_names = load_class_names(dataset_dir)

        logging.info(
            f"EvaluationPipeline: {request.actual_model.name} ({request.actual_model.framework.name})"
        )
        request.actual_model.metrics = request.actual_model.predict_dataset(
            request.actual_test_dataset
        )

        report_dir = (
            Path(request.eval_path)
            / request.actual_test_dataset_path.get("name")
            / request.actual_model_path.get("name")
            / "classification_report"
        )

        save_full_report(
            report_dir=report_dir,
            metrics_dict=request.actual_model.metrics,
            class_names=class_names,
        )

        return super().handle(request)
