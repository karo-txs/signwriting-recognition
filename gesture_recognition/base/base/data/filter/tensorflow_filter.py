from collections import defaultdict
import tensorflow as tf


def filter_dataset_by_str_classes(dataset, classes_to_keep: list):
    """
    The function `filter_dataset_by_str_classes` filters a dataset based on a list of classes to keep
    using TensorFlow operations.

    Args:
        dataset: The `dataset` parameter is typically a collection of data that you want to filter based
    on certain criteria. It could be a dataset of images, text, or any other type of data that you are
    working with in your machine learning or data processing task.
        classes_to_keep (list): The `classes_to_keep` parameter is a list of classes that you want to keep
    in the dataset. The function `filter_dataset_by_str_classes` takes a dataset and filters it based on
    the classes provided in the `classes_to_keep` list.

    Returns:
        The function `filter_dataset_by_str_classes` returns a filtered dataset based on the classes
    specified in the `classes_to_keep` list. The dataset is filtered using a filter function that checks
    if the label of each data point is in the `classes_to_keep` list. If the label is in the list, the
    data point is included in the filtered dataset.
    """
    classes_to_keep_set = tf.constant(classes_to_keep)

    def filter_fn(features, label):
        is_in_classes = tf.reduce_any(tf.equal(label, classes_to_keep_set))
        return is_in_classes
    filtered_dataset = dataset.filter(filter_fn)
    return filtered_dataset


def limit_samples_per_class(dataset: tf.data.Dataset, limit: int) -> tf.data.Dataset:
    class_counter = defaultdict(int)

    def filter_fn(features, label):
        class_counter[label.numpy()] += 1

        return class_counter[label.numpy()] <= limit

    limited_dataset = dataset.filter(lambda features, label: tf.py_function(
        func=filter_fn, inp=[features, label], Tout=tf.bool))

    return limited_dataset
