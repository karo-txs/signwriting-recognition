from core.data.interfaces.hand_landmark_detector import HandLandmarkDetector
from core.utils.landmark_functions import draw_landmarks_on_image
from core.utils.path_functions import get_internal_asset
from mediapipe.tasks import python
import mediapipe as mp



class Mediapipe(HandLandmarkDetector):

    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=get_internal_asset("hand_landmarker.task")
        )
        options = python.vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=2
        )
        self.detector = python.vision.HandLandmarker.create_from_options(options)

    def _find_highest_hand(self, detection_result):
        """
        The function `_find_highest_hand` returns the index of the hand with the highest wrist position in a
        given detection result.
        """
        min_y = float("inf")
        highest_hand_index = -1

        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            wrist_y = hand_landmarks[0].y
            if wrist_y < min_y:
                min_y = wrist_y
                highest_hand_index = i

        return highest_hand_index

    def detect_from_chunks(
        self,
        image_path: str,
        label: str,
        save_path: str,
        save_landmark_image: bool = True,
    ):
        """
        The function `get_highest_hand_landmark_data_from_path` processes an image to detect hand landmarks,
        retrieves data for the highest detected hand, and optionally saves a visual representation of the
        landmarks.
        """
        file_name = image_path.split("/")[-1]
        save_landmark_path = f"{save_path}/{label}/{file_name}"

        mp_image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(mp_image)

        if (
            detection_result.handedness is not None
            and len(detection_result.handedness) > 0
        ):
            selected_hand_index = self._find_highest_hand(detection_result)

            hand_landmarks = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in detection_result.hand_landmarks[selected_hand_index]
            ]
            handedness_scores = [
                h.score for h in detection_result.handedness[selected_hand_index]
            ]
            handedness_name = [
                h.category_name
                for h in detection_result.handedness[selected_hand_index]
            ]
            hand_world_landmarks = [
                [hand_landmark.x, hand_landmark.y, hand_landmark.z]
                for hand_landmark in detection_result.hand_world_landmarks[
                    selected_hand_index
                ]
            ]

            if save_landmark_image:
                draw_landmarks_on_image(
                    [detection_result.hand_landmarks[selected_hand_index]],
                    save_landmark_path,
                )

            return {
                "hand_landmark": hand_landmarks,
                "handedness": handedness_scores,
                "handedness_name": handedness_name,
                "world_hand": hand_world_landmarks,
            }
        else:
            return None
