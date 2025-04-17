from core.utils.path_functions import create_directories_for_file
import tensorflow as tf
import numpy as np


def write_map_func_float_features_and_string_label(features, label):
    label_bytes = (
        tf.strings.as_string(label)
        if isinstance(label, tf.Tensor)
        else label.encode("utf-8")
    )
    feature_dict = {
        "features": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        "label": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[
                    (
                        label_bytes.numpy()
                        if isinstance(label_bytes, tf.Tensor)
                        else label_bytes
                    )
                ]
            )
        ),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def _batch_py_write(features_batch, labels_batch):
    out = [
        write_map_func_float_features_and_string_label(f, l)
        for f, l in zip(features_batch, labels_batch)
    ]
    return np.asarray(out, dtype=object)


def save_dataset_with_tfrecord(
    dataset,
    exporter_dataset_path: str,
    batch_size: int = 1024,
    shards: int = 8,
    compression: str = "GZIP",
):
    create_directories_for_file(exporter_dataset_path)

    ds_serial = (
        dataset.batch(batch_size)
        .map(
            lambda f, l: tf.py_function(_batch_py_write, inp=[f, l], Tout=tf.string),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .unbatch()
        .prefetch(tf.data.AUTOTUNE)
    )

    for i in range(shards):
        tf.data.experimental.TFRecordWriter(
            f"{exporter_dataset_path}/{i:02d}.tfrecord", compression_type=compression
        ).write(ds_serial.shard(shards, i))
