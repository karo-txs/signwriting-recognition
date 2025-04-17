from core.data.steps.landmark_detector import mediapipe_landmark_detector


def landmark_detector_from_chunks(
    detector_name: str, image_path, label, save_path: str
):
    file_name = image_path.split("/")[-1]

    if detector_name == "mediapipe":
        landmarks = (
            mediapipe_landmark_detector.get_highest_hand_landmark_data_from_path(
                image_path,
                save_landmark_image=f"{save_path}/landmarks/{label}/{file_name}",
                save_landmark_path=True,
            )
        )

    return landmarks
