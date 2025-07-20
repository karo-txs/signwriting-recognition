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

    def handle(self, request: EvaluationPipeline) -> EvaluationPipeline:
        logging.info("EvaluationPipeline: Run Bootstrap Report")

        logging.info(
            "EvaluationPipeline: %s (%s)",
            request.actual_model.name,
            request.actual_model.framework.name,
        )
        
        report_dir = (
            Path(request.eval_path)
            / request.actual_test_dataset_path.get("name")
            / request.actual_model_path.get("name")
            / "bootstrap_report"
        )

        save_bootstrap_report(
            y_true=request.actual_model.metrics["y_true"],
            y_pred=request.actual_model.metrics["y_pred"],
            report_dir=report_dir,
            n_iterations=self.n_iterations,
            sample_frac=self.sample_frac,
            rng_seed=self.rng_seed,
        )

        return super().handle(request)
