from base.hand_gesture import landmark_generation, landmark_generation_control


def landmark_augmentation(tf_dataset, generate_methods, max_gestures,
                          save_path, save_landmark_image=True):
    generate_methods_fn = []

    for method_name in generate_methods:
        if method_name == "rotate_finger":
            generate_methods_fn.append(
                landmark_generation.augment_with_rotate_finger)
        elif method_name == "perturb_points":
            generate_methods_fn.append(
                landmark_generation.augment_with_perturb_points)

    return landmark_generation_control.create_generation_dataset(tf_dataset,
                                                                 methods=generate_methods_fn,
                                                                 max_gestures=max_gestures,
                                                                 save_landmark_image=save_landmark_image,
                                                                 save_landmark_path=f"{save_path}/augmented")
