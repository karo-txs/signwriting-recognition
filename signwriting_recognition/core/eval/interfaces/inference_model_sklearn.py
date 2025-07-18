from core.eval.interfaces.inference_model import InferenceModel
from core.utils.enums import Framework
from dataclasses import dataclass
from typing import Any, Dict
from tqdm import tqdm
import numpy as np
import time


@dataclass
class InferenceModelSklearn(InferenceModel):
    framework: Framework = Framework.SKLEARN

    def predict_dataset(self, dataset) -> Dict[str, Any]:
        """
        `dataset` deve iterar sobre `(input, label)`
        onde `input[0]` são landmarks (21×3) compatíveis com treino.
        """
        y_true, y_pred = [], []
        n_samples, total_time = 0, 0.0

        for x, lbl in tqdm(dataset, desc=f"[{self.name}] infer"):
            landmarks = (
                x[0] if isinstance(x[0], np.ndarray) else x[0].numpy()
            ).flatten()

            t0 = time.time()
            pred = int(self.model.predict([landmarks])[0])
            total_time += time.time() - t0

            y_pred.append(pred)
            y_true.append(int(lbl))
            n_samples += 1

        return self._build_metrics(y_true, y_pred, total_time, n_samples)
