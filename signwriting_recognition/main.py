import os, warnings

os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging, click, yaml
from typing import Any, Dict

from core.dtype import MultiConstructLoader

yaml.SafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    MultiConstructLoader.construct_mapping,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import mediapipe as mp

from service.mapper.map_yaml_to_config import parse_config
from core.train.train_pipeline import run_train_pipeline
from core.eval.eval_pipeline import run_eval_pipeline
from infra.logging.logging_utils import setup_logging
from core.data.data_pipeline import run_data_pipeline


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Carrega o arquivo YAML utilizando o loader customizado que lida com chaves duplicadas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=MultiConstructLoader)


@click.command()
@click.option("--config-path", default="../experiments/configs/data_pipeline.yaml")
def main(config_path):

    raw_cfg = load_yaml_config(config_path)
    cfg = parse_config(raw_cfg)

    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Config carregada com sucesso.")

    for pipe_cfg in cfg.data_pipelines:
        logger.info(f"DataPipeline: {pipe_cfg.original_path}")
        run_data_pipeline(pipe_cfg)

    for pipe_cfg in cfg.train_pipeline:
        setup_logging(experiment_path=pipe_cfg.experiment_path)

        name = pipe_cfg.experiment_path.split("/")[-1]

        tlogger = logging.getLogger(f"train.{name}")
        tlogger.info(f"TrainPipeline: {name}")

        run_train_pipeline(pipe_cfg)

    for pipe_cfg in cfg.evaluation_pipeline:
        logger.info(f"EvalPipeline: {pipe_cfg.eval_path}")

        run_eval_pipeline(pipe_cfg)


if __name__ == "__main__":
    main()
