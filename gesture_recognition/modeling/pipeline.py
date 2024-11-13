# fmt: off
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
import logging
import argparse
from training import train, hyper
from evaluation import eval, summarize
from models import landmark_detector
from base.base.utilities import config
from base.base.utilities import dict_utils
from data import load, save, normalization, augmentation, utils, sampler
# fmt: on


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main(yaml_config):
    pipeline = config.load_yaml_config(yaml_config)
    experiment_name = yaml_config.split("\\")[-1].split(".")[0]
    base_data_path = pipeline.get("base_data_path")
    experiment_path = f"""{pipeline.get("experiment_path")}/{experiment_name}"""
    package_cfg = pipeline.get("package")
    training_cfg = pipeline.get("training")
    evaluation_cfg = pipeline.get("evaluation")

    data_processed = {"train": [], "val": [], "test": []}

    if "package" not in pipeline.get("skip_steps", []):
        logging.info("Data processing")
        for dataset in package_cfg:
            logging.info("Processing dataset %s", dataset.get("dataset"))
            source_path = f"""{base_data_path}/{dataset.get("dataset")}/images"""
            save_path = f"""{experiment_path}/dataset/{dataset.get("use")}/{dataset.get("dataset")}"""
            os.makedirs(save_path, exist_ok=True)

            if dataset.get("sample", None):
                logging.info("Creating sample")
                target_path = f"{save_path}/sample"
                sampler.create_sample(
                    source_path, target_path, sample_size=dataset.get("sample"))
                source_path = target_path

            logging.info("Processing step: Capture")
            dataset_chunk = load.load_chunk_data(source_path)
            for i, chunk_data in enumerate(dataset_chunk):
                hand_data_labels = []
                landmark_dict = {}

                for image_path in chunk_data.keys():
                    landmarks = landmark_detector.landmark_detector_from_chunks(detector_name=pipeline.get("landmark_detector"),
                                                                                image_path=image_path,
                                                                                label=chunk_data[image_path],
                                                                                save_path=save_path)
                    if landmarks:
                        hand_data_labels.append(chunk_data[image_path])
                        dict_utils.add_to_dict(landmark_dict, landmarks)

                tf_dataset = load.load_tf_dataset_from_dict(
                    landmark_dict, hand_data_labels)

                if dataset.get("preprocess", None):
                    logging.info("Processing step: Normalization")
                    tf_dataset = normalization.dataset_landmark_normalization(
                        tf_dataset, save_path)

                if dataset.get("augmentation", None):
                    logging.info("Processing step: Augmentation")
                    tf_dataset = augmentation.landmark_augmentation(tf_dataset,
                                                                    generate_methods=dataset.get(
                                                                        "augmentation")["methods"],
                                                                    max_gestures=dataset.get(
                                                                        "augmentation")["factor"],
                                                                    save_path=save_path)

                save.save_chunk_dataset(tf_dataset,
                                        exporter_dataset_path=save_path,
                                        chunk_name=i)

            save.save_concatenated_chunk_dataset(
                save_path, utils.get_dataset_name(dataset))
            data_processed[dataset.get("use")].append((save_path,
                                                       utils.get_dataset_name(dataset)))

        logging.info("Data package")
        for split_datasets in data_processed.keys():
            save.save_ready_dataset(experiment_path=f"""{experiment_path}/dataset/{split_datasets}""",
                                    datasets_path=data_processed[split_datasets],
                                    dataset_param_to_class_filter=data_processed["test"][0][0])

    if "train" not in pipeline.get("skip_steps", []):
        logging.info("Model Training")
        for model in training_cfg:
            # if "hyper" not in pipeline.get("skip_steps", []):
            #     hyper.tuner_model(experiment_path, model)
            train.train_model(experiment_path, model)

    if "eval" not in pipeline.get("skip_steps", []):
        logging.info("Evaluation")
        for model in training_cfg:
            for eval_method in evaluation_cfg.get("methods"):
                eval.evaluate_model(experiment_path, model,
                                    eval_method, base_data_path, package_cfg)

    if "summary" not in pipeline.get("skip_steps", []):
        logging.info("Summary Results")
        for eval_method in evaluation_cfg.get("methods"):
            summarize.summarize(experiment_path, training_cfg, eval_method)

    shutil.copyfile(yaml_config, f'{experiment_path}/pipeline.yml')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run data pipeline with configuration from YAML.')
    parser.add_argument('yaml_config', type=str,
                        help='Path to the YAML configuration file')
    args = parser.parse_args()

    main(args.yaml_config)
