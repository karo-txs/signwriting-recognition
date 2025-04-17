from core.utils.performance_functions import measure_time
from core.utils.dict_functions import add_to_dict
from core.utils.landmark_functions import (
    draw_landmarks_on_image,
)
import tensorflow as tf
import numpy as np
import random
import math


@measure_time
def landmark_augmentation(
    tf_dataset, generate_methods, max_gestures, save_path, save_landmark_image=True
):
    generate_methods_fn = []

    for method_name in generate_methods:
        if method_name == "rotate_finger":
            generate_methods_fn.append(augment_with_rotate_finger)
        elif method_name == "perturb_points":
            generate_methods_fn.append(augment_with_perturb_points)

    return create_generation_dataset(
        tf_dataset,
        methods=generate_methods_fn,
        max_gestures=max_gestures,
        save_landmark_image=save_landmark_image,
        save_landmark_path=save_path,
    )


def augment_with_rotate_finger(hand_data, max_samples=1):
    max_variations = np.arange(1.0, 10.0, 0.1)
    random.shuffle(max_variations)
    max_variations = max_variations[:max_samples]

    finger_indices = [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18, 19, 20]]

    samples = []
    for max_variation in max_variations:
        for finger in finger_indices:
            new_hand_data = hand_data.copy()
            new_hand_data["hand_landmark"] = rotate_finger(
                new_hand_data["hand_landmark"], finger, max_variation
            )
            new_hand_data["world_hand"] = rotate_finger(
                new_hand_data["world_hand"], finger, max_variation
            )
            samples.append(new_hand_data)
    return samples


def augment_with_perturb_points(hand_data, max_samples=1):
    max_variations = np.arange(0.001, 0.005, 0.001)
    random.shuffle(max_variations)
    max_variations = max_variations[:max_samples]

    samples = []
    for max_variation in max_variations:
        new_hand_data = hand_data.copy()
        new_hand_data["hand_landmark"] = perturb_points(
            new_hand_data["hand_landmark"], max_variation
        )
        new_hand_data["world_hand"] = perturb_points(
            new_hand_data["world_hand"], max_variation
        )
        samples.append(new_hand_data)
    return samples


def rotate_finger(hand_landmarks, finger_indices, max_angle_degrees=5):
    base_idx, mid_idx, tip_idx = finger_indices
    base = hand_landmarks[base_idx]
    mid = hand_landmarks[mid_idx]
    tip = hand_landmarks[tip_idx]

    # Calcula o vetor base-tip e base-mid
    base_to_mid = mid - base
    base_to_tip = tip - base

    # Gera um ângulo aleatório dentro do limite especificado
    angle = tf.random.uniform([], -max_angle_degrees, max_angle_degrees) * math.pi / 180

    # Matriz de rotação no plano XY
    cos_val = tf.cos(angle)
    sin_val = tf.sin(angle)
    rotation_matrix = tf.stack(
        [[cos_val, -sin_val, 0.0], [sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]]
    )

    # Rotaciona os vetores
    rotated_base_to_mid = tf.tensordot(base_to_mid, rotation_matrix, axes=1)
    rotated_base_to_tip = tf.tensordot(base_to_tip, rotation_matrix, axes=1)

    # Atualiza as posições do mid e tip
    new_mid = base + rotated_base_to_mid
    new_tip = base + rotated_base_to_tip

    # Substitui os landmarks originais pelos rotacionados
    hand_landmarks_updated = tf.tensor_scatter_nd_update(
        hand_landmarks, indices=[[mid_idx], [tip_idx]], updates=[new_mid, new_tip]
    )

    return hand_landmarks_updated


def perturb_points(hand_data, perturbation_range=0.02):
    perturbation = tf.random.uniform(
        tf.shape(hand_data), -perturbation_range, perturbation_range
    )
    perturbed_hand = hand_data + perturbation
    return perturbed_hand


@measure_time
def create_generation_dataset(
    dataset,
    methods: list,
    max_gestures=5,
    save_landmark_image: bool = True,
    save_landmark_path: str = "",
):
    hand_data_dict = {}
    hand_data_label = []

    for feature, label in dataset:
        variations = generate_variations_per_method(feature, methods, max_gestures)

        for variation in variations:
            add_to_dict(hand_data_dict, variation)
            hand_data_label.append(label)

    hand_ds = tf.data.Dataset.from_tensor_slices(hand_data_dict)
    label_ds = tf.data.Dataset.from_tensor_slices(hand_data_label)
    hand_label_ds = tf.data.Dataset.zip((hand_ds, label_ds))

    if save_landmark_image:
        for i, (data, label) in enumerate(hand_label_ds):
            label_str = label.numpy().decode("utf-8")
            draw_landmarks_on_image(
                [data["hand_landmark"]], f"{save_landmark_path}/{label_str}/{i}.png"
            )

    return hand_label_ds


def generate_variations_per_method(landmarks, methods: list, max_gestures=5):
    variations = []
    for generate_method in methods:
        variations.extend(generate_method(landmarks, max_gestures))
    return variations
