from core.utils.tf_data_functions import (
    convert_labels_to_int,
    filter_dataset_by_str_classes,
    read_map_fn_with_str_label,
    read_tfrecord,
    save_dataset_with_tfrecord,
    write_map_func_float_features_and_int_label,
)
from core.utils.counter_functions import count_sample_per_class
from typing import Any, Dict, List
import tensorflow as tf
import json
import os


def prepare_dataset(
    datasets_path: List[Dict[str, Any]],
    split: str,
    experiment_path: str,
    label_names: str = None,
):
    tf_datasets = []

    for dataset_path in datasets_path:
        path = dataset_path.get("path")
        dataset = read_tfrecord([path], read_map_fn_with_str_label, embedding_size=127)
        tf_datasets.append(dataset)

    dataset = concatenate_datasets(tf_datasets)

    if label_names:
        with open(label_names, 'r') as json_file:
            label_names_list = list(json.load(json_file)["classes"].keys())
        dataset = filter_dataset_by_str_classes(dataset, classes_to_keep=label_names_list)

    dataset_with_int_labels, label_map = convert_labels_to_int(dataset)

    save_dataset_with_tfrecord(
        dataset_with_int_labels,
        write_map_fn=write_map_func_float_features_and_int_label,
        exporter_dataset_path=f"{experiment_path}/{split}.tfrecord",
    )

    get_data_info(experiment_path, dataset, split=split)

    report_path = os.path.join(experiment_path, f"{split}_labels.json")
    with open(report_path, "w") as json_file:
        json.dump(label_map, json_file, indent=4)


def get_data_info(output_path: str, dataset: any, split: str, dtype=str):
    count_classes = count_sample_per_class(dataset, dtype=dtype)

    data_info = {"total_samples": sum(count_classes.values()), "classes": count_classes}

    json_path = os.path.join(output_path, f"{split}_info.json")
    with open(json_path, "w") as json_file:
        json.dump(data_info, json_file, indent=4)

    return json_path


def concatenate_datasets(datasets: list) -> tf.data.Dataset:
    concatenated_dataset = datasets[0]

    for dataset in datasets[1:]:
        concatenated_dataset = concatenated_dataset.concatenate(dataset)

    return concatenated_dataset
