from core.utils.landmark_functions import draw_landmarks_on_image
from core.utils.performance_functions import measure_time
import math, tensorflow as tf
import numpy as np
import random


FINGER_INDICES = [(2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]
ROTATE_VARIATIONS = np.arange(1.0, 10.0, 0.1)
PERTURB_VARIATIONS = np.arange(0.001, 0.005, 0.001)


@measure_time
def landmark_augmentation(
    tf_dataset, generate_methods, max_gestures, save_path, save_landmark_image=True
):
    methods_map = {
        "rotate_finger": augment_with_rotate_finger,
        "perturb_points": augment_with_perturb_points,
    }
    methods = [methods_map[name] for name in generate_methods if name in methods_map]
    return create_generation_dataset(
        tf_dataset,
        methods=methods,
        max_gestures=max_gestures,
        save_landmark_image=save_landmark_image,
        save_landmark_path=save_path,
    )


def augment_with_rotate_finger(hand_data, max_samples=1):
    angles = random.sample(
        list(ROTATE_VARIATIONS), k=min(max_samples, len(ROTATE_VARIATIONS))
    )
    samples = []
    for angle in angles:
        for finger in FINGER_INDICES:
            hd = hand_data.copy()
            hd["hand_landmark"] = rotate_finger(hd["hand_landmark"], finger, angle)
            hd["world_hand"] = rotate_finger(hd["world_hand"], finger, angle)
            samples.append(hd)
    return samples


def augment_with_perturb_points(hand_data, max_samples=1):
    deltas = random.sample(
        list(PERTURB_VARIATIONS), k=min(max_samples, len(PERTURB_VARIATIONS))
    )
    return [
        {
            **hand_data,
            "hand_landmark": perturb_points(hand_data["hand_landmark"], delta),
            "world_hand": perturb_points(hand_data["world_hand"], delta),
        }
        for delta in deltas
    ]


def rotate_finger(hand_landmarks, finger_indices, max_angle_degrees=5):
    base_idx, mid_idx, tip_idx = finger_indices
    base, mid, tip = (
        hand_landmarks[base_idx],
        hand_landmarks[mid_idx],
        hand_landmarks[tip_idx],
    )

    angle = tf.random.uniform([], -max_angle_degrees, max_angle_degrees) * math.pi / 180
    c, s = tf.cos(angle), tf.sin(angle)
    R = tf.stack([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    b2m = mid - base
    b2t = tip - base
    new_mid = base + tf.tensordot(b2m, R, axes=1)
    new_tip = base + tf.tensordot(b2t, R, axes=1)

    return tf.tensor_scatter_nd_update(
        hand_landmarks,
        indices=[[mid_idx], [tip_idx]],
        updates=[new_mid, new_tip],
    )


def perturb_points(hand_data, perturbation_range=0.02):
    noise = tf.random.uniform(
        tf.shape(hand_data), -perturbation_range, perturbation_range
    )
    return hand_data + noise


@measure_time
def create_generation_dataset(
    dataset,
    methods: list,
    max_gestures=5,
    save_landmark_image: bool = True,
    save_landmark_path: str = "",
):
    data_list = []
    labels = []
    for features, label in dataset:
        for var in generate_variations_per_method(features, methods, max_gestures):
            data_list.append(var)
            labels.append(label)

    if not data_list:
        return tf.data.Dataset.from_tensor_slices(([], []))

    keys = data_list[0].keys()
    features_dict = {k: tf.stack([d[k] for d in data_list], axis=0) for k in keys}

    hand_ds = tf.data.Dataset.from_tensor_slices(features_dict)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    hand_label_ds = tf.data.Dataset.zip((hand_ds, label_ds))

    if save_landmark_image:
        for i, (data, label) in enumerate(hand_label_ds):
            lbl = label.numpy().decode("utf-8")
            draw_landmarks_on_image(
                [data["hand_landmark"]], f"{save_landmark_path}/{lbl}/{i}.png"
            )

    return hand_label_ds


def generate_variations_per_method(landmarks, methods: list, max_gestures=5):
    # Flatten das listas de variações
    return [v for m in methods for v in m(landmarks, max_gestures)]
