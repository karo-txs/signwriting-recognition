from core.eval.interfaces.inference_model import InferenceModel
from core.utils.enums import Framework 
from dataclasses import dataclass, field
from typing import Any, Dict, List
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path


@dataclass
class InferenceModelTFLite(InferenceModel):
    """
    Wrapper para modelos TensorFlow-Lite (*.tflite*) que imita o comportamento
    do InferenceModelTensorflow.
    """

    model_path: Path
    framework: Framework = Framework.TFLITE

    interpreter: tf.lite.Interpreter = field(init=False, repr=False)
    input_details: List[Dict[str, Any]] = field(init=False, repr=False)
    output_details: List[Dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self):
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_dataset(self, dataset) -> Dict[str, Any]:
        """
        Percorre um tf.data.Dataset que produz (input, label) e devolve
        as mesmas métricas de _build_metrics() usadas no wrapper TF.
        """
        y_true, y_pred = [], []
        n_samples, total_time = 0, 0.0

        for x, lbl in tqdm(dataset, desc=f"[{self.name}] infer"):
            x_in = self._format_input(x)

            t0 = time.time()
            logits = self._run_interpreter(x_in)
            total_time += time.time() - t0

            y_pred.append(int(np.argmax(logits, axis=1)[0]))
            y_true.append(int(lbl))
            n_samples += 1

        return self._build_metrics(y_true, y_pred, total_time, n_samples)

    def _format_input(self, x):
        """
        Reproduz a lógica usada no modelo TF (dict → lista de tensores,
        tupla → lista de tensores, tensor rank-1 → expand_dims, etc.).
        Retorna *numpy arrays* (Interpreter não aceita tf.Tensor).
        """
        if isinstance(x, dict):
            xs = [
                tf.expand_dims(x["hand_landmarks_input"], 0),
                tf.expand_dims(x["handness_input"], 0),
                tf.expand_dims(x["landmarks_word_input"], 0),
            ]
            return [t.numpy() for t in xs]

        elif isinstance(x, tuple):
            return [tf.expand_dims(t, 0).numpy() for t in x]

        elif len(x.shape) == 1:
            return tf.expand_dims(x, 0).numpy()

        else:
            return x.numpy() if tf.is_tensor(x) else x

    def _run_interpreter(self, x_in):
        """
        Faz a inferência TFLite:
          • Ajusta o shape do tensor se necessário
          • Define tensor(es) de entrada
          • Chama invoke()
          • Devolve o logits
        """
        if isinstance(x_in, list):  # multi-entrada
            assert len(x_in) == len(
                self.input_details
            ), "N° entradas do .tflite ≠ entradas fornecidas"
            for det, arr in zip(self.input_details, x_in):
                self._maybe_resize(det, arr.shape)
                self.interpreter.set_tensor(det["index"], arr)
        else:
            det = self.input_details[0]
            self._maybe_resize(det, x_in.shape)
            self.interpreter.set_tensor(det["index"], x_in)

        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

    def _maybe_resize(self, detail, new_shape):
        """
        Se o modelo foi convertido com batch size fixo (por ex. [1, 42]),
        redimensiona o tensor de entrada para acomodar qualquer shape que
        surgir no dataset.
        """
        if list(detail["shape"]) != list(new_shape):
            self.interpreter.resize_tensor_input(detail["index"], new_shape)
            self.interpreter.allocate_tensors()
