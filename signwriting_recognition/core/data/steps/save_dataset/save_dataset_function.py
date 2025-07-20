import os, json, logging, concurrent.futures
from collections import Counter
import tensorflow as tf
import numpy as np


def _example_from_tensors(hand_lm, world_lm, handedness, label):
    flat = tf.concat([tf.reshape(hand_lm, [-1]), tf.reshape(world_lm, [-1])], 0)
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "features": tf.train.Feature(float_list=tf.train.FloatList(value=flat)),
                "handedness": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[handedness])
                ),
                "label": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.strings.as_string(label).numpy()]
                    )
                ),
            }
        )
    ).SerializeToString()


def _write_chunk_worker(idx, feats, labels, root, ds_name):
    file_name = f"chunk_{idx:05d}.tfrecord.gz"
    out_path = os.path.join(root, ds_name, file_name)

    ds_bin = tf.data.Dataset.from_tensor_slices((feats, labels)).map(
        lambda f, l: tf.py_function(
            _example_from_tensors,
            inp=[f["hand_landmark"], f["world_hand"], f["handedness"], l],
            Tout=tf.string,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    tf.io.gfile.makedirs(os.path.dirname(out_path))
    tf.data.experimental.TFRecordWriter(out_path, compression_type="GZIP").write(ds_bin)

    labels_np = labels.numpy()

    kind = labels_np.dtype.kind
    if kind in ("S", "a"):  # bytes
        labels_str = np.char.decode(labels_np, "utf-8")
    else:  # 'U', 'i', 'f', 'O', …
        labels_str = labels_np.astype(str)

    uniques, counts = np.unique(labels_str, return_counts=True)
    return dict(zip(uniques.tolist(), counts.tolist()))


def export_chunks(
    tf_dataset: tf.data.Dataset,
    chunk_size: int,
    exporter_dataset_path: str,
    dataset_name: str,
    max_workers: int | None = os.cpu_count(),
    executor_type: str = "process",
):
    batched = tf_dataset.batch(chunk_size).prefetch(tf.data.AUTOTUNE)

    Executor = (
        concurrent.futures.ProcessPoolExecutor
        if executor_type == "process"
        else concurrent.futures.ThreadPoolExecutor
    )

    global_counter: Counter[str] = Counter()

    with Executor(max_workers=max_workers) as pool:
        futures = []
        for idx, (feat_b, lab_b) in enumerate(batched):
            futures.append(
                pool.submit(
                    _write_chunk_worker,
                    idx,
                    feat_b,
                    lab_b,
                    exporter_dataset_path,
                    dataset_name,
                )
            )

        for fut in concurrent.futures.as_completed(futures):
            local = fut.result()
            global_counter.update({str(k): v for k, v in local.items()})

    root = os.path.join(exporter_dataset_path, dataset_name)
    os.makedirs(root, exist_ok=True)
    info = {
        "total_samples": int(sum(global_counter.values())),
        "classes": dict(global_counter),
    }
    with open(os.path.join(root, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    logging.info("[Save] info.json escrito.")
