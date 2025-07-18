from core.eval.interfaces.eval_metrics import EvalMetrics
from dataclasses import dataclass, field
from core.utils.enums import Framework
from typing import Any, Dict
import numpy as np


@dataclass
class InferenceModel:
    """Contém o objeto modelo e metadados mínimos."""

    name: str
    framework: Framework = field(default=None)
    model: Any = field(default=None, repr=False)
    metrics: EvalMetrics = field(default=None)

    def predict_dataset(self, dataset) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_metrics(self, y_true, y_pred, total_time, n_samples) -> Dict[str, Any]:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        return {
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "average_inference_time": total_time / max(n_samples, 1e-9),
            "throughput": n_samples / max(total_time, 1e-9),
        }
