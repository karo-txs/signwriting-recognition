from core.pipeline import Config, DataPipeline, EvaluationPipeline, TrainPipeline
from typing import Any, Dict


def ensure_list(item):
    """
    Garante que o item seja uma lista. Se não for, empacota-o numa lista.
    """
    if item is None:
        return []
    if isinstance(item, list):
        return item
    return [item]


def parse_config(raw_config: Dict[str, Any]) -> Config:
    """
    Transforma o dicionário carregado do YAML em uma instância do dataclass Config.
    """
    landmark_detector = raw_config.get("landmark_detector")

    raw_data_pipelines = ensure_list(raw_config.get("data_pipeline"))
    data_pipelines = []
    for dp in raw_data_pipelines:
        data_pipelines.append(
            DataPipeline(
                original_path=dp.get("original_path"),
                target_path=dp.get("target_path"),
                save_intermediate_steps=dp.get("save_intermediate_steps"),
                steps=dp.get("steps", []),
            )
        )

    raw_train_pipeline = ensure_list(raw_config.get("train_pipeline"))
    train_pipelines = []
    for params in raw_train_pipeline:
        train_pipelines.append(
            TrainPipeline(
                train_dataset_paths=params.get("train_dataset"),
                val_dataset_paths=params.get("val_dataset"),
                test_dataset_paths=params.get("test_dataset"),
                models_params=params.get("models"),
                experiment_path=params.get("experiment_path"),
                label_names=params.get("label_names"),
            )
        )

    raw_eval_pipeline = ensure_list(raw_config.get("eval_pipeline"))
    eval_pipelines = []
    for params in raw_eval_pipeline:
        eval_pipelines.append(
            EvaluationPipeline(
                eval_path=params.get("eval_path"),
                label_names=params.get("label_names"),
                models_path=params.get("models_path"),
                test_dataset_paths=params.get("test_dataset_paths")
            )
        )

    return Config(
        landmark_detector=landmark_detector,
        data_pipelines=data_pipelines,
        train_pipeline=train_pipelines,
        evaluation_pipeline=eval_pipelines,
    )
