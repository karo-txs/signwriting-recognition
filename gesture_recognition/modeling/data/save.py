from base.base.data import tensorflow_create, tensorflow_write, tensorflow_read, tensorflow_filter, tensorflow_convert
from base.hand_gesture import landmark_transforms
from base.base.data import tensorflow_write
from base.base.utilities import getter
from data import utils
import json
import os


def save_chunk_dataset(tf_dataset, exporter_dataset_path, chunk_name):
    tf_dataset_concat = landmark_transforms.apply_transform(
        tf_dataset, landmark_transforms.concat_dataset)

    tensorflow_write.save_dataset_with_tfrecord(tf_dataset_concat,
                                                write_map_fn=tensorflow_write.write_map_func_float_features_and_string_label,
                                                exporter_dataset_path=f"{exporter_dataset_path}/chunks/{chunk_name}.tfrecord")


def save_concatenated_chunk_dataset(save_path, dataset_name):
    chunk_concat_dataset = tensorflow_create.create_concatenated_dataset_from_folder(f"{save_path}/chunks",
                                                                                     tensorflow_read.read_map_fn_with_str_label)

    tensorflow_write.save_dataset_with_tfrecord(chunk_concat_dataset,
                                                write_map_fn=tensorflow_write.write_map_func_float_features_and_string_label,
                                                exporter_dataset_path=f"{save_path}/tfrecords/{dataset_name}.tfrecord")


def save_ready_dataset(experiment_path: str,
                       datasets_path: list,
                       dataset_param_to_class_filter: str):
    tf_datasets = []

    for dataset_path in datasets_path:
        dataset = tensorflow_read.read_tfrecord([f"{dataset_path[0]}/tfrecords/{dataset_path[1]}.tfrecord"],
                                                tensorflow_read.read_map_fn_with_str_label,
                                                embedding_size=127)
        utils.get_data_info(dataset_path[0], dataset)
        tf_datasets.append(dataset)

    dataset = tensorflow_create.concatenate_datasets(tf_datasets)

    if dataset_param_to_class_filter:
        label_names = getter.get_all_folder_names(
            f"{dataset_param_to_class_filter}/landmarks")
        dataset = tensorflow_filter.filter_dataset_by_str_classes(
            dataset, classes_to_keep=label_names)

    dataset_with_int_labels, label_map = tensorflow_convert.convert_labels_to_int(
        dataset)

    split = experiment_path.split("/")[-1]
    tensorflow_write.save_dataset_with_tfrecord(dataset_with_int_labels,
                                                write_map_fn=tensorflow_write.write_map_func_float_features_and_int_label,
                                                exporter_dataset_path=f"{experiment_path}/{split}.tfrecord")

    utils.get_data_info(experiment_path, dataset)

    report_path = os.path.join(experiment_path, "labels.json")
    with open(report_path, 'w') as json_file:
        json.dump(label_map, json_file, indent=4)
