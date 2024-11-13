from tqdm import tqdm
import numpy as np
import time


def evaluate_test_data(model, test_case):
    """
    Avalia um modelo sklearn usando o dataset de teste.
    """
    y_true_list = []
    y_pred_list = []
    total_inference_time = 0
    total_samples = 0

    for input, label in tqdm(test_case):
        if isinstance(input[0], np.ndarray):
            landmarks = input[0].flatten()
        else:
            landmarks = input[0].numpy().flatten()
        start_time = time.time()
        y_pred = model.predict([landmarks])[0]
        inference_time = time.time() - start_time

        total_inference_time += inference_time
        total_samples += 1

        y_true_list.append(label)
        y_pred_list.append(y_pred)

    average_inference_time = total_inference_time / total_samples
    throughput = total_samples / total_inference_time

    return {
        "y_true": np.array(y_true_list),
        "y_pred": np.array(y_pred_list),
        "average_inference_time": average_inference_time,
        "throughput": throughput
    }
