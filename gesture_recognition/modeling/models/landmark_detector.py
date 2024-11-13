from base.hand_gesture import landmark_collection


def landmark_detector_from_chunks(detector_name: str, image_path, label, save_path: str):
    file_name = image_path.split("/")[-1]

    if detector_name == "mediapipe":
        landmarks = landmark_detector_with_mediapipe(image_path,
                                                     save_landmark_path=f"{save_path}/landmarks/{label}/{file_name}")

    return landmarks


def landmark_detector_with_mediapipe(image_path, save_landmark_path, save_landmark_image=True):
    return landmark_collection.get_highest_hand_landmark_data_from_path(image_path,
                                                                        save_landmark_image=save_landmark_image,
                                                                        save_landmark_path=save_landmark_path)
