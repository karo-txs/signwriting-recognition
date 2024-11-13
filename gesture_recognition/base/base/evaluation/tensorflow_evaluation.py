import tensorflow as tf
from tqdm import tqdm
import numpy as np
import time


def evaluate_test_data(model, test_case):
    y_true_list = []
    y_pred_list = []
    total_inference_time = 0
    total_samples = 0

    for input, label in tqdm(test_case):
        if not isinstance(input, tuple) and len(input.shape) == 1:
            input_data = tf.expand_dims(input, axis=0)
        elif isinstance(input, tuple):
            input_data = [np.expand_dims(input_tensor, axis=0)
                          for input_tensor in input]

        start_time = time.time()
        batch_predictions = model.predict(input_data, verbose=0)
        inference_time = time.time() - start_time

        y_pred = np.argmax(batch_predictions, axis=1)[0]

        total_inference_time += inference_time
        total_samples += 1

        y_true_list.append(label)
        y_pred_list.append(y_pred)

    average_inference_time = total_inference_time / total_samples
    throughput = total_samples / total_inference_time

    return {"y_true": np.array(y_true_list),
            "y_pred": np.array(y_pred_list),
            "average_inference_time": average_inference_time,
            "throughput": throughput}
