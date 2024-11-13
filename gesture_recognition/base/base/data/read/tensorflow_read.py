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
            label_val = label.numpy().decode('utf-8')
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
                label = label.decode('utf-8')

        unique_classes.add(label)

    return len(unique_classes), unique_classes


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
    dataset = dataset.map(
        lambda proto: parse_tfrecord_fn(proto, embedding_size))

    dataset = dataset.shuffle(buffer_size=count_samples(dataset))
    return dataset


def read_map_fn(proto, embedding_size=127):
    """
    This function reads and parses a serialized example from a TFRecord file containing features and a
    label.

    Args:
        proto: `proto` is a protocol buffer containing serialized data that needs to be parsed. In this
    function, it is used as an input to parse the features and labels from the serialized data.
        embedding_size: The `embedding_size` parameter specifies the size of the embedding vector for the
    features in the input data. In the provided function `read_map_fn`, this parameter is used to define
    the length of the fixed-length feature 'features' in the TFRecord dataset. By default, the
    `embedding_size`. Defaults to 127

    Returns:
        The function `read_map_fn` is returning the parsed features from the input protocol buffer
    `proto`. Specifically, it returns the 'features' and 'label' extracted from the parsed protocol
    buffer as a tuple.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([embedding_size], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    return parsed_features['features'], parsed_features['label']


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
        'features': tf.io.FixedLenFeature([embedding_size], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    return parsed_features['features'], parsed_features['label']


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
        'features': tf.io.FixedLenFeature([embedding_size], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)

    screen_landmarks = tf.reshape(parsed_features['features'][:63], (21, 3))
    handedness = tf.reshape(parsed_features['features'][63:64], (1, 1))
    world_landmarks = tf.reshape(parsed_features['features'][64:127], (21, 3))

    return (screen_landmarks, handedness, world_landmarks), parsed_features['label']


def read_map_fn_unconcat_with_str_label(proto, embedding_size=127):
    """
    This function reads a serialized example containing features and a label, then parses and reshapes
    the features before returning them along with the label.

    Args:
        proto: `proto` is a protocol buffer containing serialized data that needs to be parsed to extract
    specific features and labels.
        embedding_size: The `embedding_size` parameter specifies the size of the embedding for each
    feature in the input data. In this specific function `read_map_fn_unconcat_with_str_label`, the
    `embedding_size` is set to 127 by default. This means that each feature in the input data is
    expected to have. Defaults to 127

    Returns:
        (screen_landmarks, handedness, world_landmarks), parsed_features['label']
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([embedding_size], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    screen_landmarks = tf.reshape(parsed_features['features'][:63], (21, 3))
    handedness = tf.reshape(parsed_features['features'][63:64], (1, 1))
    world_landmarks = tf.reshape(parsed_features['features'][64:127], (21, 3))
    return (screen_landmarks, handedness, world_landmarks), parsed_features['label']


def read_map_fn_only_hand_landmarks(proto, embedding_size=127):
    """
    This function reads hand landmarks from a protobuf file and returns the landmarks along with their
    corresponding label.

    Args:
        proto: `proto` is a protocol buffer containing serialized data that needs to be parsed to extract
    specific features and labels.
        embedding_size: The `embedding_size` parameter specifies the size of the embedding vector for each
    data point in the input features. In this case, it is set to 127. Defaults to 127

    Returns:
        The function `read_map_fn_only_hand_landmarks` returns the screen landmarks (hand landmarks) as a
    21x3 tensor and the label as an integer.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature([embedding_size], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    screen_landmarks = tf.reshape(parsed_features['features'][:63], (21, 3))
    return screen_landmarks, parsed_features['label']
