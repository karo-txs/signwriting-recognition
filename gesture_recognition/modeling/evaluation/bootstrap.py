from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from base.base.evaluation import tensorflow_evaluation, sklearn_evaluation
from tensorflow.keras.models import load_model
from sklearn.utils import resample
import tensorflow as tf
from data import load
import numpy as np
import joblib
import json
import os


def recursive_to_numpy(element):
    """
    Recursivamente converte tensores para arrays NumPy em tuplas ou listas aninhadas.
    """
    if isinstance(element, tf.Tensor):
        return element.numpy()
    elif isinstance(element, (tuple, list)):
        return type(element)(recursive_to_numpy(e) for e in element)
    else:
        return element


def numpy_to_serializable(obj):
    """
    Função para converter arrays NumPy em listas para serialização JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Converte o array NumPy para lista
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def bootstrap_evaluate(experiment_path, training_cfg, n_iterations=10, sample_size=None):
    model_path = f"{experiment_path}/model/{training_cfg.get('model')}/best_model"
    if os.path.isdir(model_path):
        output_result_path = f"""{experiment_path}/evaluate/{training_cfg.get("model")}/bootstrap"""
        os.makedirs(output_result_path, exist_ok=True)

        if training_cfg.get("framework") == "tensorflow":
            model = load_model(model_path)
        elif training_cfg.get("framework") == "sklearn":
            model = joblib.load(
                f"{model_path}/{training_cfg.get('model')}.joblib")

        # Carrega o dataset
        dataset_test = load.load_ready_dataset(
            f"{experiment_path}/dataset/test/test.tfrecord")

        y_true_all = []
        y_pred_all = []
        inference_times = []
        throughputs = []

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        # Converte o dataset inteiro em uma lista de numpy arrays para poder usar no resample
        numpy_samples = [recursive_to_numpy(sample) for sample in dataset_test]

        # Se sample_size for None, define como 40% do tamanho total do dataset
        if sample_size is None:
            sample_size = int(0.4 * len(numpy_samples))

        for i in range(n_iterations):
            # Bootstrap amostral usando resample com o tamanho de sample_size
            bootstrap_sample = resample(
                numpy_samples, replace=True, n_samples=sample_size, random_state=i)

            # Avaliação do modelo
            if training_cfg.get("framework") == "tensorflow":
                model_result = tensorflow_evaluation.evaluate_test_data(
                    model, bootstrap_sample)
            elif training_cfg.get("framework") == "sklearn":
                model_result = sklearn_evaluation.evaluate_test_data(
                    model, bootstrap_sample)

            y_true = model_result["y_true"]
            y_pred = model_result["y_pred"]

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            inference_times.append(model_result["average_inference_time"])
            throughputs.append(model_result["throughput"])

            # Calculando métricas de classificação para cada iteração
            accuracy_list.append(accuracy_score(y_true, y_pred))
            precision_list.append(precision_score(
                y_true, y_pred, average='weighted'))
            recall_list.append(recall_score(
                y_true, y_pred, average='weighted'))
            f1_list.append(f1_score(y_true, y_pred, average='weighted'))

            # Salvando os resultados de cada iteração
            iteration_result_path = os.path.join(
                output_result_path, f"bootstrap_iteration_{i + 1}.json")
            with open(iteration_result_path, 'w') as f:
                json.dump({
                    # Convertendo para listas
                    "y_true": numpy_to_serializable(np.array(y_true)),
                    # Convertendo para listas
                    "y_pred": numpy_to_serializable(np.array(y_pred)),
                    "accuracy": accuracy_list[-1],
                    "precision": precision_list[-1],
                    "recall": recall_list[-1],
                    "f1_score": f1_list[-1],
                    "average_inference_time": model_result["average_inference_time"],
                    "throughput": model_result["throughput"]
                }, f, indent=4)

        # Calcula as médias, mínimos e máximos das métricas de classificação e outros valores
        avg_inference_time = np.mean(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)

        avg_throughput = np.mean(throughputs)
        min_throughput = np.min(throughputs)
        max_throughput = np.max(throughputs)

        avg_accuracy = np.mean(accuracy_list)
        min_accuracy = np.min(accuracy_list)
        max_accuracy = np.max(accuracy_list)

        avg_precision = np.mean(precision_list)
        min_precision = np.min(precision_list)
        max_precision = np.max(precision_list)

        avg_recall = np.mean(recall_list)
        min_recall = np.min(recall_list)
        max_recall = np.max(recall_list)

        avg_f1 = np.mean(f1_list)
        min_f1 = np.min(f1_list)
        max_f1 = np.max(f1_list)

        # Salva os resultados agregados com média, mínimo e máximo
        aggregated_results_path = os.path.join(
            output_result_path, "bootstrap_aggregated_results.json")
        with open(aggregated_results_path, 'w') as f:
            json.dump({
                "average_inference_time": avg_inference_time,
                "min_inference_time": min_inference_time,
                "max_inference_time": max_inference_time,
                "average_throughput": avg_throughput,
                "min_throughput": min_throughput,
                "max_throughput": max_throughput,
                "average_accuracy": avg_accuracy,
                "min_accuracy": min_accuracy,
                "max_accuracy": max_accuracy,
                "average_precision": avg_precision,
                "min_precision": min_precision,
                "max_precision": max_precision,
                "average_recall": avg_recall,
                "min_recall": min_recall,
                "max_recall": max_recall,
                "average_f1_score": avg_f1,
                "min_f1_score": min_f1,
                "max_f1_score": max_f1
            }, f, indent=4)
