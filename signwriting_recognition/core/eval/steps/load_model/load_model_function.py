from core.eval.interfaces.inference_model_tensorflow import InferenceModelTensorflow
from core.eval.interfaces.inference_model_sklearn import InferenceModelSklearn
from core.eval.interfaces.inference_model import InferenceModel
from pathlib import Path
from typing import List
import tensorflow as tf
import logging
import joblib


def _find_savedmodel_dir(model_root: Path) -> Path | None:
    """
    Procura por um diretório que contenha `saved_model.pb` dentro de `model_root`
    (primeiro nível).  Retorna o caminho ou None.
    """
    # 1) próprio diretório
    if (model_root / "saved_model.pb").exists():
        return model_root

    # 2) sub-diretórios imediatos (ex.: "best_model/", "ckpt-42/")
    for child in model_root.iterdir():
        if child.is_dir() and (child / "saved_model.pb").exists():
            return child
    return None


def load_models_from_experiment(experiment_path: str) -> List[InferenceModel]:
    """
    Percorre `<experiment_path>/models/<modelo>/` e tenta carregar:
      • primeiro arquivo *.keras
      • senão arquivo *.joblib
      • senão diretório contendo saved_model.pb  (TensorFlow SavedModel)

    Retorna lista de InferenceModel.
    """
    models_dir = Path(experiment_path) / "models"
    if not models_dir.exists():
        logging.warning("Diretório de modelos não existe: %s", models_dir)
        return []

    loaded: List[InferenceModel] = []

    for model_root in sorted(models_dir.iterdir()):
        if not model_root.is_dir():
            continue

        keras_files = list(model_root.glob("*.keras"))
        if keras_files:
            path = keras_files[0]
            try:
                model = tf.keras.models.load_model(path, compile=False)
                loaded.append(InferenceModelTensorflow(name=model_root.name, model=model))
                logging.info("Modelo TensorFlow (*.keras) carregado: %s", path)
                continue
            except Exception as e:
                logging.error("Falha ao carregar %s: %s", path, e)

        joblib_files = list(model_root.glob("*.joblib"))
        if joblib_files:
            path = joblib_files[0]
            try:
                model = joblib.load(path)
                loaded.append(InferenceModelSklearn(name=model_root.name, model=model))
                logging.info("Modelo sklearn carregado: %s", path)
                continue
            except Exception as e:
                logging.error("Falha ao carregar %s: %s", path, e)

        sm_dir = _find_savedmodel_dir(model_root)
        if sm_dir is not None:
            try:
                model = tf.keras.models.load_model(sm_dir, compile=False)
                loaded.append(InferenceModelTensorflow(name=model_root.name, model=model))
                logging.info("SavedModel carregado: %s", sm_dir)
            except Exception as e:
                logging.error("Falha ao carregar SavedModel em %s: %s", sm_dir, e)
        else:
            logging.warning(
                "Nenhum artefato reconhecido (*.keras, *.joblib, saved_model.pb) em %s",
                model_root,
            )

    return loaded