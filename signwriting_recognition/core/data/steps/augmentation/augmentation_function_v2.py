from core.utils.landmark_functions import draw_landmarks_on_image
from core.utils.performance_functions import measure_time
from core.utils.dict_functions import add_to_dict
import tensorflow as tf
import functools
import math
import time
import os


_FINGER_IDXS = tf.constant(
    [[2, 3, 4],
     [6, 7, 8],
     [10, 11, 12],
     [14, 15, 16],
     [18, 19, 20]],
    dtype=tf.int32
)


def _tf_rotate_finger(hand_xyz, finger_triplet, angle_deg):
    """Rotaciona um dedo em torno do ponto base (plano XY)."""
    base_idx, mid_idx, tip_idx = tf.unstack(finger_triplet)
    base   = hand_xyz[base_idx]
    mid    = hand_xyz[mid_idx]
    tip    = hand_xyz[tip_idx]

    angle  = angle_deg * math.pi / 180.0
    cos_a, sin_a = tf.cos(angle), tf.sin(angle)
    rot = tf.stack([[cos_a, -sin_a, 0.0],
                    [sin_a,  cos_a, 0.0],
                    [0.0,    0.0,   1.0]])

    v_mid = tf.tensordot(mid - base, rot, axes=1)
    v_tip = tf.tensordot(tip - base, rot, axes=1)

    updates = tf.stack([base + v_mid, base + v_tip])
    idxs    = tf.stack([mid_idx, tip_idx])
    return tf.tensor_scatter_nd_update(hand_xyz, tf.expand_dims(idxs, 1), updates)


def _rotate_fingers_all(hand_xyz, max_angle_deg):
    """Aplica a rotação a cada dedo (5 dedos) com um mesmo ângulo."""
    def _one_finger(finger):
        return _tf_rotate_finger(hand_xyz, finger, max_angle_deg)

    # retorna lista [5] de versões; stack + concat para (5, 21, 3)
    return tf.vectorized_map(_one_finger, _FINGER_IDXS)


def _augment_rotate(sample, n_variations):
    # amostras de ângulo uniformes em [-10, 10] deg
    angles = tf.random.uniform([n_variations], -10.0, 10.0)
    hand   = sample["hand_landmark"]
    world  = sample["world_hand"]

    def _one(angle):
        return {
            "hand_landmark": _rotate_fingers_all(hand, angle),
            "world_hand"   : _rotate_fingers_all(world, angle)
        }

    # shape final: (n_variations * 5, 21, 3)
    aug = tf.vectorized_map(_one, angles)
    aug = tf.nest.map_structure(
        lambda x: tf.reshape(x, [-1] + x.shape.as_list()[2:]), aug
    )
    return aug


def _augment_perturb(sample, n_variations):
    noise = tf.random.uniform(
        [n_variations, 21, 3],
        -0.005, 0.005, dtype=tf.float32
    )
    hand = sample["hand_landmark"]
    world = sample["world_hand"]

    perturbed = {
        "hand_landmark": hand + noise,
        "world_hand": world + noise
    }
    # reshape para (n_variations, 21, 3)
    return perturbed


@measure_time
def landmark_augmentation(
    tf_dataset,
    generate_methods,
    max_gestures,
    save_path,
    save_landmark_image=True,
):
    """
    Retorna um dataset com (feature_dict, label) após aplicar as
    variações listadas em `generate_methods`. Interface preservada.
    """
    if save_landmark_image:
        os.makedirs(save_path, exist_ok=True)

    # mapeia string -> função
    method_table = {
        "rotate_finger": lambda s: _augment_rotate(s, max_gestures),
        "perturb_points": lambda s: _augment_perturb(s, max_gestures),
    }
    methods_fn = [method_table[m] for m in generate_methods]

    # aplica cada método e concatena resultados
    def _apply_methods(feature, label):
        # saída original para manter semântica
        variations = {"hand_landmark": tf.expand_dims(feature["hand_landmark"], 0),
                      "world_hand":   tf.expand_dims(feature["world_hand"], 0)}
        for fn in methods_fn:
            aug = fn(feature)
            variations = tf.nest.map_structure(lambda a, b: tf.concat([a, b], axis=0),
                                               variations, aug)

        labels = tf.repeat(label, tf.shape(variations["hand_landmark"])[0])
        return tf.data.Dataset.from_tensor_slices((variations, labels))

    augmented_ds = tf_dataset.flat_map(_apply_methods)

    if save_landmark_image:
        save_func = functools.partial(_save_img_py, base_path=save_path)

        def _map_fn(feature, label):
            tf.py_function(
                func=save_func,
                inp=[feature["hand_landmark"], label],
                Tout=[]
            )
            return feature, label

        augmented_ds = augmented_ds.map(
            _map_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    return augmented_ds.prefetch(tf.data.AUTOTUNE)


def _save_img_py(hand_landmark, label_tensor, base_path):
    """
    Efeito colateral: grava imagem dos landmarks.
    hand_landmark  : np.ndarray (21,3)
    label_tensor   : 0‑D np.bytes_ ou np.str_
    """
    label_np  = label_tensor.numpy()
    label_str = (
        label_np.decode("utf-8")
        if isinstance(label_np, (bytes, bytearray))
        else str(label_np)
    )

    subdir = os.path.join(base_path, label_str)
    os.makedirs(subdir, exist_ok=True)

    fname = f"{int(time.time()*1e6)}.png"
    draw_landmarks_on_image([hand_landmark], os.path.join(subdir, fname))