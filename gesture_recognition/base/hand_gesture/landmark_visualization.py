import base.base.data.write.image_write as image_write
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np


def draw_landmarks_on_image(hand_landmarks_list, output_image_path: str):
    """
    The function `draw_landmarks_on_image` takes a list of hand landmarks and draws them on a blank
    image.

    Args:
        hand_landmarks_list: The `hand_landmarks_list` parameter is a list containing the landmarks of a
    hand. Each element in the list represents the landmarks of a single hand and can be in different
    formats, either as a list of tuples `(x, y, z)` or as a list of `landmark_pb2
        output_image_path (str): The `output_image_path` parameter is a string that represents the file
    path where the annotated image with landmarks will be saved after drawing the landmarks on it.
    """
    blank_image = np.full((480, 700, 3), 255, dtype=np.uint8)
    annotated_image = np.copy(blank_image)
    for _, hand_landmarks in enumerate(hand_landmarks_list):
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        try:
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
        except:
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in hand_landmarks
            ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    image_write.save_image_with_cv2(
        output_path=output_image_path, image=annotated_image)

    return annotated_image
