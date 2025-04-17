from core.utils.performance_functions import measure_time
import tensorflow as tf


@measure_time
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