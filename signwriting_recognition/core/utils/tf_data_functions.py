from core.utils.landmark_functions import draw_landmarks_on_image
from core.utils.path_functions import create_directories_for_file
from core.utils.performance_functions import measure_time
import tensorflow as tf
import glob


def apply_transform(dataset, transform_func):
    """
    The function `apply_transform` takes a dataset and a transformation function, applies the
    transformation function to each element in the dataset, and returns the transformed dataset.
    """
    dataset = dataset.map(
        map_func=lambda feature, label: (transform_func(feature), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


@measure_time
def apply_transform_and_report(
    dataset,
    transform_func,
    save_landmark_image: bool = True,
    save_landmark_path: str = "",
):
    """
    The function `apply_transform` applies a specified transformation function to a dataset and
    optionally saves landmark images.
    """
    dataset = apply_transform(dataset, transform_func)

    if save_landmark_image:
        for i, (data, label) in enumerate(dataset):
            label_str = label.numpy().decode("utf-8")
            draw_landmarks_on_image(
                [data["hand_landmark"]], f"{save_landmark_path}/{label_str}/{i}.png"
            )
    return dataset


def concat_dataset(hand_data):
    """
    The function `concat_dataset` concatenates three different types of hand data into a single tensor.
    """
    concatenated_data = tf.concat(
        [
            tf.reshape(hand_data["hand_landmark"], [-1]),
            tf.reshape(hand_data["handedness"], [-1]),
            tf.reshape(hand_data["world_hand"], [-1]),
        ],
        axis=0,
    )
    return concatenated_data


@measure_time
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
    create_directories_for_file(exporter_dataset_path)
    tf_dataset = dataset.map(
        lambda features, label: tf.py_function(
            func=write_map_fn, inp=[features, label], Tout=tf.string
        )
    )

    writer = tf.data.experimental.TFRecordWriter(exporter_dataset_path)
    writer.write(tf_dataset)


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
    label_bytes = (
        tf.strings.as_string(label)
        if isinstance(label, tf.Tensor)
        else label.encode("utf-8")
    )

    feature_dict = {
        "features": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        "label": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[
                    (
                        label_bytes.numpy()
                        if isinstance(label_bytes, tf.Tensor)
                        else label_bytes
                    )
                ]
            )
        ),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example_proto.SerializeToString()


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
        "features": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


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
        raise FileNotFoundError("No TFRecord files found in the specified folder.")

    return read_tfrecord(tfrecord_files, read_map_fn)


def read_tfrecord(dataset_path: str, parse_tfrecord_fn, embedding_size=127):
    """
    The function `read_tfrecord` reads a TFRecord dataset, applies a parsing function to each record,
    shuffles the dataset, and returns the resulting dataset.

    Args:
        dataset_path (str): The `dataset_path` parameter is a string that represents the file path to the
    TFRecord dataset that you want to read and process. This function reads the TFRecord dataset from
    the specified path and applies a parsing function (`parse_tfrecord_fn`) to each record in the
    dataset.
        parse_tfrecord_fn: The `parse_tfrecord_fn` is a function that is used to parse the serialized data
    stored in a TFRecord file. This function takes the serialized protocol buffer (proto) as input and
    returns the parsed data in a format that can be used by your model. It typically involves decoding
    the serialized data
        embedding_size: The `embedding_size` parameter specifies the size of the embedding vector that
    will be used for each record in the dataset. It is typically a hyperparameter that you can adjust
    based on the requirements of your machine learning model. In this case, the default value is set to
    127, but you can. Defaults to 127

    Returns:
        The function `read_tfrecord` is returning a TensorFlow dataset that has been created from a
    TFRecord dataset located at the `dataset_path`. The dataset is processed using the
    `parse_tfrecord_fn` function with the specified `embedding_size`. The dataset is then shuffled using
    a buffer size determined by the number of samples in the dataset before being returned.
    """
    dataset = tf.data.TFRecordDataset(dataset_path)
    dataset = dataset.map(lambda proto: parse_tfrecord_fn(proto, embedding_size))

    dataset = dataset.shuffle(buffer_size=count_samples(dataset))
    return dataset


def count_samples(dataset):
    """
    The function `count_samples` calculates the size of a dataset by iterating through its elements.

    Args:
        dataset: The `count_samples` function takes a dataset as input and counts the number of samples in
    that dataset. The dataset is expected to be a collection of data points, where each data point is a
    tuple containing the data and its corresponding features.

    Returns:
        The function `count_samples` returns the size of the dataset, which is the total number of samples
    in the dataset.
    """
    dataset_size = 0
    for data, feature in dataset:
        dataset_size += 1

    return dataset_size


def read_map_fn_with_str_label(proto, embedding_size=127):
    """
    This function reads a TFRecord file containing features and a string label.

    Args:
        proto: The `proto` parameter is typically a serialized example in TensorFlow, which contains the
    features and label information that we want to extract and process. This function is designed to
    read a serialized example containing a feature vector and a string label, parse it using the
    specified feature description, and return the extracted features and
        embedding_size: The `embedding_size` parameter specifies the size of the embedding vector for the
    features in the input data. In this case, it is set to a default value of 127, but you can adjust it
    based on the requirements of your model or dataset. Defaults to 127

    Returns:
        The function `read_map_fn_with_str_label` returns the parsed features 'features' and 'label' from
    the input protocol buffer `proto`.
    """
    feature_description = {
        "features": tf.io.FixedLenFeature([embedding_size], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    return parsed_features["features"], parsed_features["label"]


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
        unique_labels = sorted(
            set(label.numpy().decode("utf-8") for _, label in dataset)
        )
        class_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    class_mapping_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(class_mapping.keys())),
            values=tf.constant(list(class_mapping.values()), dtype=tf.int64),
        ),
        default_value=-1,
    )

    def map_fn(features, label):
        label_int = class_mapping_table.lookup(label)
        return features, label_int

    dataset = dataset.map(map_fn)
    return dataset, class_mapping


def read_map_fn_unconcat(proto, embedding_size=127):
    """
    This function reads and parses a serialized example containing features and a label, then reshapes
    the features into screen landmarks, handedness, and world landmarks before returning them along with
    the label.

    Args:
        proto: The `proto` parameter is a protocol buffer containing serialized data that needs to be
    parsed. In this function, it is used to parse the features and labels from the serialized data.
        embedding_size: The `embedding_size` parameter specifies the size of the embedding vector for each
    data point in the input features. In this case, it is set to a default value of 127, but you can
    adjust it based on the requirements of your model or dataset. Defaults to 127

    Returns:
        The function `read_map_fn_unconcat` is returning a tuple containing three elements:
    `screen_landmarks`, `handedness`, `world_landmarks`, and the parsed label. `screen_landmarks` is a
    tensor of shape (21, 3) containing the first 63 elements of the 'features' tensor, `handedness` is a
    tensor of shape (1,
    """
    feature_description = {
        "features": tf.io.FixedLenFeature([embedding_size], tf.float32),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)

    screen_landmarks = tf.reshape(parsed_features["features"][:63], (21, 3))
    handedness = tf.reshape(parsed_features["features"][63:64], (1, 1))
    world_landmarks = tf.reshape(parsed_features["features"][64:127], (21, 3))

    return (screen_landmarks, handedness, world_landmarks), parsed_features["label"]
