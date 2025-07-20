from core.train.intefaces.tf_model import TFModel
from core.train.intefaces.model import Model
import tensorflow as tf
import os


def save_model(model: Model):

    if isinstance(model, TFModel):
        model.builded_model
        tflite_path = os.path.join(model.models_path, "best_model.tflite")

        converter = tf.lite.TFLiteConverter.from_saved_model(
            f"{model.models_path}/best_model"
        )
        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"Modelo salvo em: {tflite_path}")
