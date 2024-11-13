from base.hand_gesture import landmark_normalization, landmark_transforms


def dataset_landmark_normalization(tf_dataset, save_path, save_landmark_image=True):
    def normalize(hand_data):
        hand_data['hand_landmark'] = landmark_normalization.apply_all_normalizations(
            hand_data['hand_landmark'])
        hand_data['world_hand'] = landmark_normalization.apply_all_normalizations(
            hand_data['world_hand'])
        return hand_data

    tf_dataset = landmark_transforms.apply_transform_and_report(tf_dataset,
                                                                normalize,
                                                                save_landmark_image=save_landmark_image,
                                                                save_landmark_path=f"{save_path}/normalized_landmarks")

    return tf_dataset
