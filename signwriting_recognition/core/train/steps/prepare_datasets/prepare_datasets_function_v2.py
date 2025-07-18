import os, json, glob, tensorflow as tf
from collections import defaultdict
from typing import Any, Dict, List
from core.utils.tf_data_functions import (
    read_map_fn_with_str_label,
    write_map_func_float_features_and_int_label,
    filter_dataset_by_str_classes,
    convert_labels_to_int,
)
import logging

AUTOTUNE   = tf.data.AUTOTUNE
SHUFFLE_BUF = 5_000

# ------------------------------------------------------------------
# 1) Leitura eficiente ------------------------------------------------
# ------------------------------------------------------------------

def build_streaming_dataset(folders: List[str], parse_fn):
    """
    Faz streaming paralelo de TODOS os .tfrecord em várias pastas.
    """
    # 1. monta padrões "*.tfrecord"
    patterns = [os.path.join(folder, "*.tfrecord") for folder in folders]

    # 2. lista arquivos em paralelo
    files_ds = tf.data.Dataset.from_tensor_slices(patterns) \
                              .interleave(tf.data.Dataset.list_files,
                                          cycle_length=AUTOTUNE,
                                          num_parallel_calls=AUTOTUNE)

    # 3. lê cada arquivo e aplica parse_fn
    ds = files_ds.interleave(
            lambda fname: tf.data.TFRecordDataset(fname)
                                   .map(parse_fn, num_parallel_calls=AUTOTUNE),
            cycle_length=AUTOTUNE,
            num_parallel_calls=AUTOTUNE)

    return ds.shuffle(SHUFFLE_BUF).prefetch(AUTOTUNE)

# ------------------------------------------------------------------
# 2) Escrita em shards ------------------------------------------------
# ------------------------------------------------------------------

def _serialize_example(features, label):
    # **usa a sua função original para manter o formato**
    return write_map_func_float_features_and_int_label(features, label)

def write_sharded(dataset: tf.data.Dataset,
                  out_dir: str,
                  max_samples_per_shard: int = 1_000):
    """
    Grava <dataset> em vários arquivos TFRecord dentro de <out_dir>.
    Retorna um dicionário {classe: contagem}.
    """
    tf.io.gfile.makedirs(out_dir)

    shard_idx, n_in_shard = 0, 0
    writer = tf.io.TFRecordWriter(
        os.path.join(out_dir, f"{shard_idx:05d}.tfrecord"))

    class_counter = defaultdict(int)

    for features, label in dataset:
        writer.write(_serialize_example(features, label))
        n_in_shard += 1

        lbl = int(label.numpy())
        class_counter[lbl] += 1

        if n_in_shard >= max_samples_per_shard:
            writer.close()
            shard_idx += 1
            n_in_shard = 0
            writer = tf.io.TFRecordWriter(
                os.path.join(out_dir, f"{shard_idx:05d}.tfrecord"))

    writer.close()
    return class_counter

# ------------------------------------------------------------------
# 3) Pipeline “prepare_dataset” --------------------------------------
# ------------------------------------------------------------------

def prepare_dataset(
        datasets_path: List[Dict[str, Any]],
        split: str,
        experiment_path: str,
        label_names: str | None = None,
        max_samples_per_shard: int = 5_000):

    # 1. paths → list[str]
    folders = [d["path"] for d in datasets_path]

    # 2. dataset de strings
    ds = build_streaming_dataset(folders, read_map_fn_with_str_label)

    # 3. filtra classes se preciso
    if label_names:
        with open(label_names) as f:
            keep = list(json.load(f)["classes"].keys())
        ds = filter_dataset_by_str_classes(ds, keep)

    # 4. string → int
    ds_int, label_map = convert_labels_to_int(ds)

    # 5. grava em shards
    out_dir = os.path.join(experiment_path, "split", split)
    stats   = write_sharded(ds_int, out_dir, max_samples_per_shard)

    # 6. metadados
    info = {"total_samples": int(sum(stats.values())),
            "classes": {str(k): int(v) for k, v in stats.items()}}
    with open(os.path.join(experiment_path, f"{split}_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    with open(os.path.join(experiment_path, f"{split}_labels.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    logging.info(f"[✓] {split}: {info['total_samples']} amostras → {out_dir}")
