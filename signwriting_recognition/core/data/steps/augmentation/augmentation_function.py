import tensorflow as tf, functools, math, os, time
from core.utils.landmark_functions import draw_landmarks_on_image
from core.utils.performance_functions import measure_time

_FINGER_IDXS = tf.constant(
    [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18, 19, 20]], dtype=tf.int32
)


def _tf_rotate_finger(hand_xyz, finger_triplet, angle_deg):
    base_idx, mid_idx, tip_idx = tf.unstack(finger_triplet)
    base, mid, tip = hand_xyz[base_idx], hand_xyz[mid_idx], hand_xyz[tip_idx]
    angle = angle_deg * math.pi / 180.0
    c, s = tf.cos(angle), tf.sin(angle)
    rot = tf.stack([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    v_mid = tf.tensordot(mid - base, rot, axes=1)
    v_tip = tf.tensordot(tip - base, rot, axes=1)
    updates = tf.stack([base + v_mid, base + v_tip])
    idxs = tf.stack([mid_idx, tip_idx])
    return tf.tensor_scatter_nd_update(hand_xyz, tf.expand_dims(idxs, 1), updates)


def _rotate_fingers_all(hand_xyz, angle_deg):
    return tf.vectorized_map(
        lambda f: _tf_rotate_finger(hand_xyz, f, angle_deg), _FINGER_IDXS
    )


def _augment_rotate(sample, n_variations):
    angles = tf.random.uniform([n_variations], -10.0, 10.0)
    hand, world = sample["hand_landmark"], sample["world_hand"]
    aug = {
        "hand_landmark": tf.reshape(
            tf.vectorized_map(lambda a: _rotate_fingers_all(hand, a), angles),
            [-1, 21, 3],
        ),
        "world_hand": tf.reshape(
            tf.vectorized_map(lambda a: _rotate_fingers_all(world, a), angles),
            [-1, 21, 3],
        ),
    }
    return aug


def _augment_perturb(sample, n_variations):
    noise = tf.random.uniform([n_variations, 21, 3], -0.005, 0.005)
    aug = {
        "hand_landmark": sample["hand_landmark"] + noise,
        "world_hand": sample["world_hand"] + noise,
    }
    return aug


@measure_time
def landmark_augmentation(
    tf_dataset,
    generate_methods,
    max_gestures,
    save_path,
    save_landmark_image=True,
):
    if save_landmark_image:
        os.makedirs(save_path, exist_ok=True)

    method_table = {
        "rotate_finger": lambda s: _augment_rotate(s, max_gestures),
        "perturb_points": lambda s: _augment_perturb(s, max_gestures),
    }
    methods_fn = [method_table[m] for m in generate_methods]

    def _apply_methods(feature, label):
        variations = {
            "hand_landmark": tf.expand_dims(feature["hand_landmark"], 0),
            "world_hand": tf.expand_dims(feature["world_hand"], 0),
        }
        for fn in methods_fn:
            aug = fn(feature)
            variations = tf.nest.map_structure(
                lambda a, b: tf.concat([a, b], axis=0), variations, aug
            )

        n = tf.shape(variations["hand_landmark"])[0]
        variations["handedness"] = tf.repeat(feature["handedness"], n)
        labels = tf.repeat(label, n)

        return tf.data.Dataset.from_tensor_slices((variations, labels))

    ds = tf_dataset.flat_map(_apply_methods)

    if save_landmark_image:

        def _save_img_py(hand_lm, label_tensor, base):
            lab = label_tensor.numpy()
            lab = lab.decode() if isinstance(lab, (bytes, bytearray)) else str(lab)
            sub = os.path.join(base, lab)
            os.makedirs(sub, exist_ok=True)
            name = f"{int(time.time()*1e6)}.png"
            draw_landmarks_on_image([hand_lm], os.path.join(sub, name))

        save_fn = functools.partial(_save_img_py, base=save_path)

        def _side_effect(f, l):
            tf.py_function(save_fn, [f["hand_landmark"], l], Tout=[])
            return f, l

        ds = ds.map(
            _side_effect, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )

    return ds.prefetch(tf.data.AUTOTUNE)
