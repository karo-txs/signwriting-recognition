from core.eval.interfaces.inference_model_tensorflow import InferenceModelTensorflow
from core.eval.interfaces.inference_model_sklearn import InferenceModelSklearn
from core.eval.interfaces.inference_model_tflite import InferenceModelTFLite
from core.eval.interfaces.inference_model import InferenceModel
from pathlib import Path
import logging


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


def load_models_from_experiment(models_dir: str | Path) -> InferenceModel | None:
    """
    Carrega o primeiro artefato de modelo encontrado em `models_dir`
    (arquivo ou diretório) na ordem de prioridade:
        1) *.tflite
        2) *.keras
        3) *.joblib
        4) diretório SavedModel (saved_model.pb)

    Retorna a instância de InferenceModel correspondente ou None.
    """
    root = Path(models_dir)

    if not root.exists():
        logging.warning("Caminho de modelos não existe: %s", root)
        return None

    if root.is_file():
        ext = root.suffix.lower()
        name = root.stem

        try:
            if ext == ".tflite":
                logging.info("Carregando modelo TFLite: %s", root)
                return InferenceModelTFLite(name=name, model_path=root)

            if ext == ".keras":
                logging.info("Carregando modelo Keras: %s", root)
                return InferenceModelTensorflow(name=name, model_path=root)

            if ext == ".joblib":
                logging.info("Carregando modelo sklearn: %s", root)
                return InferenceModelSklearn(name=name, model_path=root)

        except Exception as e:
            logging.error("Falha ao carregar %s: %s", root, e)
            return None

        logging.warning("Extensão de arquivo não suportada: %s", root)
        return None

    # 1) busca artefatos no nível superior
    priority_patterns = ["*.tflite", "*.keras", "*.joblib"]
    for pattern in priority_patterns:
        for file_path in root.glob(pattern):
            model = load_models_from_experiment(file_path)
            if model is not None:
                return model

    # 2) SavedModel: procura saved_model.pb no próprio dir ou em subdirs
    sm_dir = _find_savedmodel_dir(root)
    if sm_dir is not None:
        try:
            logging.info("SavedModel carregado: %s", sm_dir)
            return InferenceModelTensorflow(name=sm_dir.parent.name, model_path=sm_dir)
        except Exception as e:
            logging.error("Falha ao carregar SavedModel em %s: %s", sm_dir, e)

    # 3) procura recursivamente em subdiretórios por outros artefatos
    for child in sorted(root.iterdir()):
        if child.is_dir():
            model = load_models_from_experiment(child)
            if model is not None:
                return model

    logging.warning(
        "Nenhum artefato reconhecido (*.tflite, *.keras, *.joblib, saved_model.pb) em %s",
        root,
    )
    return None
