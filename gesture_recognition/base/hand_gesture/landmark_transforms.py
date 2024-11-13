from base.base.data.write import image_write
from base.hand_gesture.landmark_collection import detect_pose_landmarks
import base.hand_gesture.landmark_visualization as vis_mediapipe
from base.base.utilities.getter import get_internal_asset
import tensorflow as tf
import cv2


def concat_dataset(hand_data):
    """
    The function `concat_dataset` concatenates three different types of hand data into a single tensor.
    """
    concatenated_data = tf.concat([tf.reshape(hand_data['hand_landmark'], [-1]),
                                   tf.reshape(hand_data['handedness'], [-1]),
                                   tf.reshape(hand_data['world_hand'], [-1])], axis=0)
    return concatenated_data


def embedder_dataset(dataset):
    """
    The function `embedder_dataset` loads a TensorFlow model, applies it to each element in a dataset,
    and returns the transformed dataset.
    """
    embedder_model = tf.keras.models.load_model(get_internal_asset("gesture_embedder/"),
                                                custom_objects={'tf': tf},
                                                compile=False
                                                )

    return dataset.batch(1).map(
        map_func=lambda features, label: (embedder_model(features), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()


def apply_transform(dataset, transform_func):
    """
    The function `apply_transform` takes a dataset and a transformation function, applies the
    transformation function to each element in the dataset, and returns the transformed dataset.
    """
    dataset = dataset.map(map_func=lambda feature, label: (transform_func(feature), label),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def apply_transform_and_report(dataset, transform_func, save_landmark_image: bool = True, save_landmark_path: str = ""):
    """
    The function `apply_transform` applies a specified transformation function to a dataset and
    optionally saves landmark images.
    """
    dataset = apply_transform(dataset, transform_func)

    if save_landmark_image:
        for i, (data, label) in enumerate(dataset):
            label_str = label.numpy().decode('utf-8')
            vis_mediapipe.draw_landmarks_on_image(
                [data["hand_landmark"]], f"{save_landmark_path}/{label_str}/{i}.png")
    return dataset


def get_hand_bounding_box(hand_landmarks, hand_indices, image_width, image_height, scale=2):
    x_coords = [int(hand_landmarks[i].x * image_width)
                for i in hand_indices]
    y_coords = [int(hand_landmarks[i].y * image_height)
                for i in hand_indices]

    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Aplica fator de escala
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(int(x_min - width * scale), 0)
    x_max = min(int(x_max + width * scale), image_width)
    y_min = max(int(y_min - height * scale), 0)
    y_max = min(int(y_max + height * scale), image_height)

    return x_min, y_min, x_max, y_max


def crop_hand_from_pose_landmarks(image_path, output_path):
    landmarks = detect_pose_landmarks(image_path)

    try:
        if landmarks:
            landmarks = landmarks[0]
            image = cv2.imread(image_path)

            # Defina os índices dos landmarks das mãos esquerda e direita
            # pulso esquerdo, dedo mínimo, dedo indicador, polegar
            left_hand_indices = [15, 17, 19, 21]
            # pulso direito, dedo mínimo, dedo indicador, polegar
            right_hand_indices = [16, 18, 20, 22]

            image_height, image_width, _ = image.shape  # Use a imagem no formato OpenCV

            # Calcule as coordenadas médias Y para ambas as mãos
            left_hand_y_coords = [landmarks[i].y *
                                  image_height for i in left_hand_indices]
            right_hand_y_coords = [landmarks[i].y *
                                   image_height for i in right_hand_indices]

            avg_left_hand_y = sum(left_hand_y_coords) / len(left_hand_y_coords)
            avg_right_hand_y = sum(right_hand_y_coords) / \
                len(right_hand_y_coords)

            # Determine qual mão é mais alta (menor valor de Y)
            if avg_left_hand_y < avg_right_hand_y:
                hand_indices = left_hand_indices
            else:
                hand_indices = right_hand_indices

            # Calcular a caixa delimitadora para a mão
            x_min, y_min, x_max, y_max = get_hand_bounding_box(
                landmarks, hand_indices, image_width, image_height)
            cropped_image = image[y_min:y_max, x_min:x_max]

            image_write.save_image_with_cv2(
                output_path=output_path, image=cropped_image)

            return cropped_image
    except:
        return None
    return None
