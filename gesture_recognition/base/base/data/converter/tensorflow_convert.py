import tensorflow as tf


def convert_labels_to_int(dataset, class_mapping=None):
    """
    The function `convert_labels_to_int` converts string labels in a dataset to integer labels using a
    provided class mapping or creating one if not provided.

    Args:
        dataset: The `dataset` parameter is expected to be a TensorFlow dataset containing pairs of
    features and labels. The function `convert_labels_to_int` takes this dataset and converts the labels
    from string format to integer format using a provided class mapping or by creating a new mapping if
    none is provided.
        class_mapping: The `class_mapping` parameter is a dictionary that maps unique labels in the
    dataset to integer values. If the `class_mapping` parameter is not provided, the function will
    automatically generate a mapping based on the unique labels found in the dataset.

    Returns:
        The function `convert_labels_to_int` returns the updated dataset with labels converted to integers
    using the provided class mapping, as well as the class mapping dictionary that was used for the
    conversion.
    """
    if class_mapping is None:
        unique_labels = sorted(set(label.numpy().decode('utf-8')
                               for _, label in dataset))
        class_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    class_mapping_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(class_mapping.keys())),
            values=tf.constant(list(class_mapping.values()), dtype=tf.int64)
        ),
        default_value=-1
    )

    def map_fn(features, label):
        label_int = class_mapping_table.lookup(label)
        return features, label_int
    dataset = dataset.map(map_fn)
    return dataset, class_mapping
