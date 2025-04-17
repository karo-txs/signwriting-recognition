from collections import defaultdict
import tensorflow as tf


def count_sample_per_class(dataset, dtype=int):
    """
    The function `count_sample_per_class` counts the number of samples per class in a dataset.

    Args:
        dataset: The `dataset` parameter is likely a collection of data samples where each sample consists
    of a data point and its corresponding label. The function `count_sample_per_class` is designed to
    count the number of samples per class in the dataset.
        dtype: The `dtype` parameter in the `count_sample_per_class` function specifies the data type to
    use when processing the labels in the dataset. It can be either `int` or `str`. If `dtype` is set to
    `int`, the labels will be converted to integers using the `dtype

    Returns:
        The function `count_sample_per_class` returns a dictionary where the keys are the unique class
    labels found in the dataset and the values are the count of samples belonging to each class.
    """
    counter = defaultdict(int)

    for _, label in dataset:
        if dtype == str:
            label_val = label.numpy().decode("utf-8")
        else:
            label_val = dtype(label.numpy())

        counter[label_val] += 1

    return counter


def count_unique_classes(dataset):
    """
    The function `count_unique_classes` takes a dataset as input and returns the number of unique
    classes in the dataset along with the set of unique classes.

    Args:
        dataset: The `dataset` parameter is a collection of data points where each data point consists of
    a pair (feature, label). The function `count_unique_classes` iterates over this dataset to extract the
    labels and count the number of unique classes present in the dataset.

    Returns:
        The function `count_unique_classes` returns a tuple containing two elements:
    1. The number of unique classes in the dataset.
    2. A set containing the unique classes found in the dataset.
    """
    unique_classes = set()

    for _, label in dataset:
        if isinstance(label, tf.Tensor):
            label = label.numpy()
            if label.ndim > 0:
                label = tuple(label)
            else:
                label = label.item()
            if isinstance(label, bytes):
                label = label.decode("utf-8")

        unique_classes.add(label)

    return len(unique_classes), unique_classes
