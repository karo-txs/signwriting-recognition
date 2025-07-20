from core.utils.tf_data_functions import (
    read_map_fn_unconcat,
)
import tensorflow as tf
import os


AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUF = 5_000


def read_map_fn_unconcat(example_proto):
    """
    Parser que devolve:
      - feature_dict {hand_landmark, world_hand, handedness}
      - label_int    (tf.int64)
    O mesmo formato em que a etapa `prepare_dataset` gravou as shards.
    """
    spec = {
        "features": tf.io.FixedLenFeature([126], tf.float32),
        "handedness": tf.io.FixedLenFeature([1], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, spec)

    flat = parsed["features"]
    feats = {
        "hand_landmarks_input": tf.reshape(flat[:63], [21, 3]),
        "landmarks_word_input": tf.reshape(flat[63:], [21, 3]),
        "handness_input": parsed["handedness"],  # shape (1,)
    }
    return feats, parsed["label"]


def load_ready_dataset(split_dir: str, shuffle_buf: int = SHUFFLE_BUF):
    """
    split_dir  →  .../datasets/split/train   (ou val, test)
    Retorna um tf.data.Dataset pronto para treinar/avaliar.
    """
    files = tf.io.gfile.glob(os.path.join(split_dir, "*.tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"Nenhum .tfrecord.gz em {split_dir}")

    ds = (
        tf.data.TFRecordDataset(
            files, compression_type="GZIP", num_parallel_reads=AUTOTUNE
        )
        .map(read_map_fn_unconcat, num_parallel_calls=AUTOTUNE)
        .shuffle(shuffle_buf)
        .prefetch(AUTOTUNE)
    )

    return ds
