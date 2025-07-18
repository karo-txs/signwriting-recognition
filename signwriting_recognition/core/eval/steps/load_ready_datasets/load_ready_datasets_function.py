from core.utils.tf_data_functions import (
    read_map_fn_unconcat,
)
import tensorflow as tf


def load_ready_dataset(experiment_path: str):
    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(f"{experiment_path}/*.tfrecord"),
        num_parallel_reads=tf.data.AUTOTUNE,
    ).map(read_map_fn_unconcat, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
