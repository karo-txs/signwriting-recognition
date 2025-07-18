from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class EvalMetrics:
    y_true: np.ndarray
    y_pred: np.ndarray
    avg_time: float
    throughput: float

    def scores(self) -> Dict[str, float]:
        return dict(
            accuracy=accuracy_score(self.y_true, self.y_pred),
            precision=precision_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            ),
            recall=recall_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            ),
            f1=f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0),
        )