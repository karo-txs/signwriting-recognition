from base.base.utilities import folder_utils
import tensorflow as tf


def save_dataset_with_tfrecord(dataset, write_map_fn, exporter_dataset_path: str):
    """
    The function `save_dataset_with_tfrecord` saves a dataset to a TFRecord file using a provided
    mapping function.

    Args:
        dataset: The `dataset` parameter is typically a TensorFlow dataset object that contains the data
    you want to save in TFRecord format. It could be created using methods like
    `tf.data.Dataset.from_tensor_slices()`, `tf.data.Dataset.from_generator()`, or by loading data from
    files.
        write_map_fn: The `write_map_fn` parameter is a function that takes in features and labels from
    the dataset and returns a serialized string representation of the data. This function is used to
    convert the dataset into a format that can be written to a TFRecord file.
        exporter_dataset_path (str): The `exporter_dataset_path` parameter is a string that represents the
    file path where the TFRecord dataset will be saved. It is the location where the TFRecord file will
    be written to on the file system.
    """
    folder_utils.create_directories_for_file(exporter_dataset_path)
    tf_dataset = dataset.map(lambda features, label: tf.py_function(
        func=write_map_fn,
        inp=[features, label],
        Tout=tf.string
    ))

    writer = tf.data.experimental.TFRecordWriter(exporter_dataset_path)
    writer.write(tf_dataset)


def write_map_func_float_features_and_int_label(features, label):
    """
    The function `write_map_func_float_features_and_int_label` converts input features and label into a
    serialized protocol buffer string.

    Args:
        features: The `features` parameter is a list of floating-point values representing the features of
    a data point.
        label: The `label` parameter in the `write_map_func_float_features_and_int_label` function is an
    integer value representing the label associated with the features.

    Returns:
        a serialized protocol buffer message (example_proto) created using the input features and label.
    """
    feature_dict = {
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def write_map_func_float_features_and_string_label(features, label):
    """
    The function `write_map_func_float_features_and_string_label` converts float features and a string
    label into a serialized protocol buffer string.

    Args:
        features: The `features` parameter is a list of floating-point values that you want to include in
    the example.
        label: The `label` parameter in the function `write_map_func_float_features_and_string_label` is
    either a TensorFlow tensor or a string. If it is a TensorFlow tensor, it is converted to a string
    using `tf.strings.as_string`. If it is already a string, it is encoded to UTF-

    Returns:
        a serialized protocol buffer message (protobuf) created using the TensorFlow `Example` class,
    which contains the provided features as float values and the label as a string.
    """
    label_bytes = tf.strings.as_string(label) if isinstance(
        label, tf.Tensor) else label.encode('utf-8')

    feature_dict = {
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes.numpy() if isinstance(label_bytes, tf.Tensor) else label_bytes]))
    }

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))

    return example_proto.SerializeToString()
