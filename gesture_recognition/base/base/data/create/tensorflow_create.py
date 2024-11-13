from base.base.data.read import tensorflow_read
import tensorflow as tf
import glob


def create_dataset_from_dict(data_dict: dict, hand_data_label: list):
    """
    The function `create_dataset_from_dict` creates a TensorFlow dataset from a dictionary of data and a
    list of labels.

    Args:
        data_dict (dict): The `data_dict` parameter is a dictionary containing the data you want to create
    a dataset from. Each key-value pair in the dictionary represents a data point, where the key is the
    data point identifier and the value is the data itself.
        hand_data_label (list): The `hand_data_label` parameter is a list containing the labels
    corresponding to the data in the `data_dict` dictionary. Each element in `hand_data_label` should
    correspond to the label for the data at the same index in the `data_dict` dictionary.

    Returns:
        The function `create_dataset_from_dict` is returning a TensorFlow dataset that contains the data
    from the input dictionary `data_dict` paired with the corresponding labels from the
    `hand_data_label` list.
    """
    hand_ds = tf.data.Dataset.from_tensor_slices(data_dict)
    label_ds = tf.data.Dataset.from_tensor_slices(hand_data_label)
    hand_label_ds = tf.data.Dataset.zip((hand_ds, label_ds))
    return hand_label_ds


def create_concatenated_dataset_from_folder(folder_path: str, read_map_fn):
    """
    The function `create_concatenated_dataset_from_folder` reads TFRecord files from a specified folder
    and returns a TensorFlow dataset.

    Args:
        folder_path (str): The `create_concatenated_dataset_from_folder` function takes a `folder_path`
    parameter, which should be a string representing the path to the folder containing TFRecord files.
    The function reads all TFRecord files from the specified folder and creates a concatenated dataset
    using `tf.data.TFRecordDataset`.

    Returns:
        The function `create_concatenated_dataset_from_folder` returns a TensorFlow dataset created from
    TFRecord files found in the specified folder.
    """
    tfrecord_files = glob.glob(f"{folder_path}/*.tfrecord")

    if not tfrecord_files:
        raise FileNotFoundError(
            "No TFRecord files found in the specified folder.")

    return tensorflow_read.read_tfrecord(tfrecord_files, read_map_fn)


def concatenate_datasets(datasets: list) -> tf.data.Dataset:
    concatenated_dataset = datasets[0]

    for dataset in datasets[1:]:
        concatenated_dataset = concatenated_dataset.concatenate(dataset)

    return concatenated_dataset
