from base.base.data import tensorflow_read
import json
import os


def get_dataset_name(data_config):
    final_name = "landmarks"

    if data_config.get("augmentation", None):
        max_gestures = data_config.get("augmentation").get("factor", 5)
        final_name += f"_G{max_gestures}x("
        for i, method_name in enumerate(data_config.get("augmentation").get("methods", [])):
            if i == 0:
                final_name += f"{method_name}"
            else:
                final_name += f"-{method_name}"
        final_name += ")"

    if data_config.get("image_transformations", None):
        final_name += f"_T("
        for i, method_name in enumerate(data_config.get("image_transformations", [])):
            if i == 0:
                final_name += f"{method_name}"
            else:
                final_name += f"-{method_name}"
        final_name += ")"

    return final_name


def get_data_info(output_path: str, dataset, dtype=str):
    count_classes = tensorflow_read.count_sample_per_class(
        dataset, dtype=dtype)

    data_info = {
        "total_samples": sum(count_classes.values()),
        "classes": count_classes
    }

    json_path = os.path.join(output_path, f"info.json")
    with open(json_path, 'w') as json_file:
        json.dump(data_info, json_file, indent=4)

    return json_path
