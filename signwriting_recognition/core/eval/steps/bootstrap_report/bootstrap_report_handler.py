from core.eval.steps.bootstrap_report.bootstrap_functions import save_bootstrap_report
from core.pipeline import EvaluationPipeline
from core.dtype import AbstractHandler
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class BootstrapReportHandler(AbstractHandler):

    n_iterations: int = 1_000
    sample_frac: float = 0.4
    rng_seed: int | None = None

    def validate(self, request: EvaluationPipeline) -> bool:
        return bool(request.test_dataset) and bool(request.models)

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        if self.validate(request):
            logging.info("EvaluationPipeline: Run Bootstrap Report")

            for model in request.models:
                logging.info("EvaluationPipeline: %s (%s)", model.name, model.framework.name)

                report_dir = (
                    Path(request.experiment_path)
                    / "models"
                    / model.name
                    / "reports"
                    / "bootstrap_report"
                )

                save_bootstrap_report(
                    y_true=model.metrics["y_true"],
                    y_pred=model.metrics["y_pred"],
                    report_dir=report_dir,
                    n_iterations=self.n_iterations,
                    sample_frac=self.sample_frac,
                    rng_seed=self.rng_seed,
                )

        return super().handle(request)
