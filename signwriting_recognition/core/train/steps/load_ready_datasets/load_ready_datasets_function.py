from core.utils.tf_data_functions import (
    read_map_fn_unconcat,
    read_tfrecord,
)


def load_ready_dataset(file_path: str):
    dataset = read_tfrecord(
        file_path, parse_tfrecord_fn=read_map_fn_unconcat, embedding_size=127
    )
    return dataset
