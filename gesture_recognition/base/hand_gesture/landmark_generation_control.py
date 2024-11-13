import base.hand_gesture.landmark_visualization as vis_mediapipe
from base.base.utilities import dict_utils
import tensorflow as tf


def generate_variations_per_method(landmarks, methods: list, max_gestures=5):
    variations = []
    for generate_method in methods:
        variations.extend(generate_method(landmarks, max_gestures))
    return variations


def create_generation_dataset(dataset, methods: list, max_gestures=5,
                              save_landmark_image: bool = True, save_landmark_path: str = ""):
    hand_data_dict = {}
    hand_data_label = []

    for feature, label in dataset:
        variations = generate_variations_per_method(
            feature, methods, max_gestures)

        for variation in variations:
            dict_utils.add_to_dict(hand_data_dict, variation)
            hand_data_label.append(label)

    hand_ds = tf.data.Dataset.from_tensor_slices(hand_data_dict)
    label_ds = tf.data.Dataset.from_tensor_slices(hand_data_label)
    hand_label_ds = tf.data.Dataset.zip((hand_ds, label_ds))

    if save_landmark_image:
        for i, (data, label) in enumerate(hand_label_ds):
            label_str = label.numpy().decode('utf-8')
            vis_mediapipe.draw_landmarks_on_image(
                [data["hand_landmark"]], f"{save_landmark_path}/{label_str}/{i}.png")

    return hand_label_ds
