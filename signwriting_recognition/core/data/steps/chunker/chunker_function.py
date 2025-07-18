from core.utils.tf_data_functions import (
    apply_transform,
    concat_dataset,
    create_concatenated_dataset_from_folder,
    read_map_fn_with_str_label,
    read_tfrecord,
    save_dataset_with_tfrecord,
    write_map_func_float_features_and_string_label,
)
from core.utils.counter_functions import count_sample_per_class
from core.utils.performance_functions import measure_time
from core.utils import path_functions
import shutil
import json
import os


@measure_time
def load_chunk_data(dataset_raw_path: str, chunk_size: int = 3000):
    file_paths = path_functions.get_all_file_paths(dataset_raw_path)
    label_names = path_functions.get_all_folder_names(dataset_raw_path)

    relation_file_label = path_functions.get_relation_of_files_per_folder_name(
        file_paths, label_names, limit_value=None
    )
    chunk_generator = separate_dict_in_chunks(
        relation_file_label, chunk_size=chunk_size
    )

    return chunk_generator


def separate_dict_in_chunks(dict_values: dict, chunk_size: int):
    """
    The function separates a dictionary into chunks of a specified size and yields one chunk at a time when called.

    :param dict_values: A dictionary of key-value pairs that you want to separate into chunks
    :type dict_values: dict
    :param chunk_size: The `chunk_size` parameter specifies the size of each chunk into which the dictionary
    will be divided
    :type chunk_size: int
    """
    items = list(dict_values.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i : i + chunk_size])


@measure_time
def save_chunk_dataset(tf_dataset, exporter_dataset_path, dataset_name, chunk_name):
    tf_dataset_concat = apply_transform(tf_dataset, concat_dataset)

    save_dataset_with_tfrecord(
        tf_dataset_concat,
        write_map_fn=write_map_func_float_features_and_string_label,
        exporter_dataset_path=f"{exporter_dataset_path}/{dataset_name}/{chunk_name}.tfrecord",
    )


@measure_time
def save_concatenated_chunk_dataset(save_path, dataset_name):
    chunk_concat_dataset = create_concatenated_dataset_from_folder(
        f"{save_path}/chunks", read_map_fn_with_str_label
    )

    save_dataset_with_tfrecord(
        chunk_concat_dataset,
        write_map_fn=write_map_func_float_features_and_string_label,
        exporter_dataset_path=f"{save_path}/{dataset_name}.tfrecord",
    )

    shutil.rmtree(f"{save_path}/chunks")
