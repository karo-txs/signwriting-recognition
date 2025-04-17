from core.data.data_pipeline import run_signwriting_data_pipeline
from service.mapper.map_yaml_to_config import parse_config
from core.train.train_pipeline import run_train_pipeline
from core.dtype import MultiConstructLoader
from core.domain import DataPipeline
from typing import Any, Dict
import logging
import yaml
import os



yaml.SafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    MultiConstructLoader.construct_mapping,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Carrega o arquivo YAML utilizando o loader customizado que lida com chaves duplicadas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo '{filepath}' não foi encontrado.")
    with open(filepath, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=MultiConstructLoader)
    return config


def process_data_pipeline(pipeline: DataPipeline):
    print(f"Processando data_pipeline com original_path: {pipeline.original_path}")
    for step in pipeline.steps:
        for step_name, params in step.items():
            print(f"  Etapa: {step_name} com parâmetros: {params}")


def main():
    #Caminho para o arquivo YAML; ajuste conforme necessário
    config_path = "../sw-experiments/configs/data_pipeline.yaml"

    try:
        raw_config = load_yaml_config(config_path)
    except Exception as e:
        print(f"Erro ao carregar o arquivo de configuração: {e}")
        return

    config = parse_config(raw_config)

    for data_pipeline_config in config.data_pipelines:
        run_signwriting_data_pipeline(data_pipeline_config)
    
    
    # config_path = "../sw-experiments/configs/train_pipeline.yaml"

    # try:
    #     raw_config = load_yaml_config(config_path)
    # except Exception as e:
    #     print(f"Erro ao carregar o arquivo de configuração: {e}")
    #     return

    # config = parse_config(raw_config)

    # for train_pipeline_config in config.train_pipeline:
    #     run_train_pipeline(train_pipeline_config)


if __name__ == "__main__":
    main()
