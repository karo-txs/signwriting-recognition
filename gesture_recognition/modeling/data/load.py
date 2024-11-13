from base.base.data import tensorflow_create, tensorflow_read, chunk
from base.base.utilities import getter


def load_chunk_data(dataset_raw_path: str, chunk_size: int = 3000):
    file_paths = getter.get_all_file_paths(dataset_raw_path)
    label_names = getter.get_all_folder_names(dataset_raw_path)

    relation_file_label = getter.get_relation_of_files_per_folder_name(
        file_paths, label_names, limit_value=None)
    chunk_generator = chunk.separate_dict_in_chunks(
        relation_file_label, chunk_size=chunk_size)

    return chunk_generator


def load_tf_dataset_from_dict(landmark_dict, hand_data_labels):
    return tensorflow_create.create_dataset_from_dict(landmark_dict, hand_data_labels)


def load_ready_dataset(file_path: str):
    dataset = tensorflow_read.read_tfrecord(file_path,
                                            parse_tfrecord_fn=tensorflow_read.read_map_fn_unconcat,
                                            embedding_size=127)
    return dataset
