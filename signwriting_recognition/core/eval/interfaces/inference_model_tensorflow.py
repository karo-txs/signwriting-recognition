from core.eval.interfaces.inference_model import InferenceModel
from core.utils.enums import Framework
from dataclasses import dataclass
from typing import Any, Dict
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time


@dataclass
class InferenceModelTensorflow(InferenceModel):
    framework: Framework = Framework.TENSORFLOW
    
    def __post_init__(self):
        if str(self.model_path).endswith("keras"):
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
        else:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def predict_dataset(self, dataset) -> Dict[str, Any]:
        """
        `dataset` deve iterar sobre `(input, label)` – onde `input`
        pode ser Tensor ou tuple de Tensors.
        """
        y_true, y_pred = [], []
        n_samples, total_time = 0, 0.0

        for x, lbl in tqdm(dataset, desc=f"[{self.name}] infer"):
            if isinstance(x, dict):
                x_in = [
                    tf.expand_dims(x["hand_landmarks_input"], 0),
                    tf.expand_dims(x["handness_input"], 0),
                    tf.expand_dims(x["landmarks_word_input"], 0),
                ]
            elif isinstance(x, tuple):
                x_in = [tf.expand_dims(t, 0) for t in x]
            elif len(x.shape) == 1:
                x_in = tf.expand_dims(x, 0)
            else:
                x_in = x

            t0 = time.time()
            logits = self.model.predict(x_in, verbose=0)
            total_time += time.time() - t0

            y_pred.append(int(np.argmax(logits, axis=1)[0]))
            y_true.append(int(lbl))

            n_samples += 1

        return self._build_metrics(y_true, y_pred, total_time, n_samples)
