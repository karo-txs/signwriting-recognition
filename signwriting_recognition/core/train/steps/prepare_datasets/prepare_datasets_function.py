from core.utils.tf_data_functions import (
    filter_dataset_by_str_classes,
    convert_labels_to_int,
)
from collections import defaultdict
from typing import Any, Dict, List
import os, json, tensorflow as tf
import concurrent.futures
import numpy as np
import logging


AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUF = 5_000


def build_streaming_dataset(folders: List[str], parse_fn):
    """
    Faz streaming paralelo de TODOS os .tfrecord.gz em várias pastas.
    """
    # 1. padrões "*.tfrecord.gz"
    patterns = [os.path.join(folder, "*.tfrecord.gz") for folder in folders]

    files_ds = tf.data.Dataset.from_tensor_slices(patterns).interleave(
        tf.data.Dataset.list_files, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE
    )

    # 2. abre cada arquivo com compressão GZIP
    ds = files_ds.interleave(
        lambda fname: tf.data.TFRecordDataset(fname, compression_type="GZIP").map(
            parse_fn, num_parallel_calls=AUTOTUNE
        ),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
    )

    return ds.shuffle(SHUFFLE_BUF).prefetch(AUTOTUNE)


def _serialize_record(hand_lm, world_lm, handedness, label_int):
    """
    hand_lm     : (21,3) float32
    world_lm    : (21,3) float32
    handedness  : ()     float32
    label_int   : ()     int64
    """
    features_dict = {
        "hand_landmark": hand_lm,
        "world_hand": world_lm,
        "handedness": handedness,
    }
    return write_map_func_float_features_and_int_label(features_dict, label_int)


def _write_shard_worker(idx, feat_batch, lbl_batch, out_dir, opts):
    """
    • feat_batch é um dict de tensores batched
      (shape: (B, 21, 3) ou (B,)  conforme o campo)
    • lbl_batch  é tensor int64 shape (B,)
    """
    file_path = os.path.join(out_dir, f"{idx:05d}.tfrecord.gz")

    ds_parts = tf.data.Dataset.from_tensor_slices(
        (
            feat_batch["hand_landmark"],
            feat_batch["world_hand"],
            feat_batch["handedness"],
            lbl_batch,
        )
    )

    ds_bin = ds_parts.map(
        lambda h, w, hd, l: tf.py_function(
            _serialize_record, inp=[h, w, hd, l], Tout=tf.string
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    tf.data.experimental.TFRecordWriter(file_path, compression_type="GZIP").write(
        ds_bin
    )

    uniques, counts = np.unique(lbl_batch.numpy(), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniques, counts)}


def write_sharded(
    dataset: tf.data.Dataset,
    out_dir: str,
    max_samples_per_shard: int = 1_000,
    max_workers: int | None = os.cpu_count(),
):
    """
    Grava shards .tfrecord.gz em paralelo.
    """
    tf.io.gfile.makedirs(out_dir)
    opts = tf.io.TFRecordOptions(compression_type="GZIP")

    # 1) divide em lotes = shards
    batched = dataset.batch(max_samples_per_shard).prefetch(tf.data.AUTOTUNE)

    global_counter = defaultdict(int)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for idx, (feat_b, lbl_b) in enumerate(batched):
            futures.append(
                pool.submit(_write_shard_worker, idx, feat_b, lbl_b, out_dir, opts)
            )

        for fut in concurrent.futures.as_completed(futures):
            local = fut.result()
            for k, v in local.items():
                global_counter[int(k)] += int(v)

    return global_counter


def prepare_dataset(
    folders: List[str],
    split: str,
    experiment_path: str,
    label_names: str | None = None,
    max_samples_per_shard: int = 5_000,
):

    # 2. dataset de strings
    ds = build_streaming_dataset(folders, read_map_fn_with_str_label)

    ds = ds.cache()

    # 3. filtra classes se preciso
    if label_names:
        with open(label_names) as f:
            keep = list(json.load(f)["classes"].keys())
        ds = filter_dataset_by_str_classes(ds, keep)

    # 4. string → int
    ds_int, label_map = convert_labels_to_int(ds)

    # 5. grava em shards
    out_dir = os.path.join(experiment_path, split)
    stats = write_sharded(ds_int, out_dir, max_samples_per_shard)

    # 6. metadados
    info = {
        "total_samples": int(sum(stats.values())),
        "classes": {str(k): int(v) for k, v in stats.items()},
        "labels": label_map
    }
    with open(os.path.join(experiment_path, f"{split}/info.json"), "w") as f:
        json.dump(info, f, indent=4)

    logging.info(f"[✓] {split}: {info['total_samples']} amostras → {out_dir}")


def read_map_fn_with_str_label(example_proto):
    """
    -> (feature_dict, label_str)
       feature_dict = {
           "hand_landmark": (21,3) float32,
           "world_hand"  : (21,3) float32,
           "handedness"  : scalar   float32
       }
    """
    spec = {
        "features": tf.io.FixedLenFeature([126], tf.float32),
        "handedness": tf.io.FixedLenFeature([1], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, spec)

    flat = parsed["features"]
    hand = tf.reshape(flat[:63], [21, 3])
    world = tf.reshape(flat[63:], [21, 3])

    feat_dict = {
        "hand_landmark": hand,
        "world_hand": world,
        "handedness": parsed["handedness"][0],  # scalar
    }
    return feat_dict, parsed["label"]


def write_map_func_float_features_and_int_label(features, label_int):
    """
    Serializa: 126 floats (hand+world)  + handedness + label(int)
    """
    flat = tf.concat(
        [
            tf.reshape(features["hand_landmark"], [-1]),
            tf.reshape(features["world_hand"], [-1]),
        ],
        axis=0,
    )  # 126

    handed = tf.expand_dims(features["handedness"], 0)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "features": tf.train.Feature(float_list=tf.train.FloatList(value=flat)),
                "handedness": tf.train.Feature(
                    float_list=tf.train.FloatList(value=handed)
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label_int])
                ),
            }
        )
    )
    return example.SerializeToString()
