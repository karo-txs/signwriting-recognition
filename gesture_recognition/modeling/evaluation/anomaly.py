from base.base.evaluation import tensorflow_evaluation, sklearn_evaluation
from tensorflow.keras.models import load_model
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy import stats
from data import load
import numpy as np
import statistics
import joblib
import json
import os


def landmarks_anomaly(experiment_path, training_cfg):
    output_result_path = f"""{experiment_path}/evaluate/{training_cfg.get("model")}/anomaly"""
    model_dir = os.path.join(output_result_path, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"""{experiment_path}/model/{training_cfg.get("model")}/best_model"""

    if os.path.isdir(model_path):
        evaluate_with_anomalies(experiment_path, training_cfg,
                                model_dir, output_result_path, anomaly_method='grubbs')
        evaluate_with_anomalies(experiment_path, training_cfg,
                                model_dir, output_result_path, anomaly_method='euclidean')
        evaluate_with_anomalies(experiment_path, training_cfg,
                                model_dir, output_result_path, anomaly_method='pca')


def flatten_landmarks(landmarks):
    """
    Achata os landmarks de (21, 3) para um vetor de 63 elementos.
    """
    return np.array(landmarks).flatten()


def grubbs_test(data, alpha=0.05):
    """
    Realiza o teste de Grubbs para detecção de outliers em um conjunto de dados.

    Parâmetros:
    - data: uma lista ou array de números.
    - alpha: nível de significância (default: 0.05).

    Retorna:
    - Índice do outlier se detectado, ou None se não houver outlier.
    """
    n = len(data)
    mean_data = np.mean(data)
    std_dev_data = np.std(data, ddof=1)

    # Calcula o valor de Grubbs
    G_max = (max(data) - mean_data) / std_dev_data
    G_min = (mean_data - min(data)) / std_dev_data

    # Calcula o valor crítico de Grubbs
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

    # Verifica se há um outlier
    if G_max > G_crit:
        return np.argmax(data), G_max  # Outlier máximo detectado
    elif G_min > G_crit:
        return np.argmin(data), G_min  # Outlier mínimo detectado
    else:
        return None, None  # Não há outliers


def detect_anomalous_gestures_per_class(gesture_landmarks_per_class, alpha=0.05, method='grubbs'):
    """
    Detecta gestos anômalos por classe usando diferentes métodos de detecção de outliers.

    Parâmetros:
    - gesture_landmarks_per_class: dicionário contendo listas de arrays de landmarks (21, 3) para cada classe.
    - alpha: nível de significância (default: 0.05).
    - method: método de detecção de outliers ('grubbs', 'euclidean', 'pca', 'mahalanobis').

    Retorna:
    - Dicionário com índices dos gestos considerados anômalos para cada classe.
    """
    outliers_per_class = {}
    for class_name, gesture_landmarks in gesture_landmarks_per_class.items():
        if method == 'grubbs':
            # Achata os landmarks para uma lista de vetores de tamanho 63
            flattened_gestures = [flatten_landmarks(
                landmark) for landmark in gesture_landmarks]

            # Calcula a média de cada ponto achatado entre todos os gestos
            gesture_means = [np.mean(gesture)
                             for gesture in flattened_gestures]

            # Aplica o teste de Grubbs para detecção de outliers
            outlier_index, G_value = grubbs_test(gesture_means, alpha)

            if outlier_index is not None:
                outliers_per_class[class_name] = [outlier_index]
            else:
                outliers_per_class[class_name] = []

        elif method == 'euclidean':
            # Calcula a distância euclidiana média dos gestos em relação ao centroide
            flattened_gestures = [flatten_landmarks(
                landmark) for landmark in gesture_landmarks]
            centroid = np.mean(flattened_gestures, axis=0)
            distances = [np.linalg.norm(gesture - centroid)
                         for gesture in flattened_gestures]

            # Define um limiar para detectar outliers
            threshold = np.mean(distances) + alpha * np.std(distances)
            outliers = [i for i, distance in enumerate(
                distances) if distance > threshold]

            outliers_per_class[class_name] = outliers

        elif method == 'pca':
            # Reduz a dimensionalidade dos gestos com PCA e detecta outliers
            flattened_gestures = [flatten_landmarks(
                landmark) for landmark in gesture_landmarks]
            if len(flattened_gestures) > 2:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(flattened_gestures)

                # Calcula a distância euclidiana no espaço de componentes principais
                centroid = np.mean(reduced_data, axis=0)
                distances = [np.linalg.norm(gesture - centroid)
                             for gesture in reduced_data]

                # Define um limiar para detectar outliers
                threshold = np.mean(distances) + alpha * np.std(distances)
                outliers = [i for i, distance in enumerate(
                    distances) if distance > threshold]
            else:
                outliers = []
            outliers_per_class[class_name] = outliers

        else:
            raise ValueError(
                "Método de detecção de outliers desconhecido. Escolha entre 'grubbs', 'euclidean', 'pca', 'mahalanobis'.")

    return outliers_per_class


# Função para avaliar o modelo de classificação com detecção de anomalias
def evaluate_with_anomalies(experiment_path, training_cfg, model_dir, output_dir, anomaly_method='grubbs'):
    """
    Combina a avaliação do modelo de classificação com a detecção de anomalias.

    Parâmetros:
    - experiment_path: caminho do experimento.
    - training_cfg: configuração de treinamento.
    - model_dir: diretório dos modelos de detecção de anomalias.
    - output_dir: diretório de saída para salvar os resultados.
    - anomaly_method: método de detecção de anomalias ('grubbs', 'euclidean', 'pca', 'mahalanobis').
    """
    model_path = f"{experiment_path}/model/{training_cfg.get('model')}/best_model"

    # Carrega o dataset de teste
    dataset_test = load.load_ready_dataset(
        f"{experiment_path}/dataset/test/test.tfrecord")
    if training_cfg.get("framework") == "tensorflow":
        model = load_model(model_path)
        model_result = tensorflow_evaluation.evaluate_test_data(
            model, dataset_test)
    elif training_cfg.get("framework") == "sklearn":
        model = joblib.load(f"{model_path}/{training_cfg.get('model')}.joblib")
        model_result = sklearn_evaluation.evaluate_test_data(
            model, dataset_test)

    y_true = model_result["y_true"]
    y_pred = model_result["y_pred"]

    # Organiza landmarks por classe
    gesture_landmarks_per_class = defaultdict(list)
    for i, (landmark_data, class_name) in enumerate(zip(dataset_test.batch(1), y_true)):
        class_name = f"{class_name}"
        landmarks_np = landmark_data[0][0].numpy()
        gesture_landmarks_per_class[class_name].append(landmarks_np)

    # Detecta anomalias por classe
    outliers_per_class = detect_anomalous_gestures_per_class(
        gesture_landmarks_per_class, method=anomaly_method)

    anomaly_report = defaultdict(
        lambda: {"total": 0, "anomalies": 0, "error_model": 0, "error_anomalies": 0})

    for class_name, gesture_landmarks in gesture_landmarks_per_class.items():
        total_samples = len(gesture_landmarks)
        anomaly_report[class_name]["total"] = total_samples

        outliers = outliers_per_class.get(class_name, [])
        anomaly_report[class_name]["anomalies"] = len(outliers)

        for i in range(total_samples):
            if y_true[i] != y_pred[i]:
                anomaly_report[class_name]["error_model"] += 1
                if i in outliers:
                    anomaly_report[class_name]["error_anomalies"] += 1

    # Gera o relatório de anomalias
    error_classes = {}
    anomaly_classes = {}
    error_anomaly_classes = {}

    error_counts = []
    anomaly_counts = []
    error_anomaly_counts = []

    for class_name, data in anomaly_report.items():
        if data["total"] > 0:
            data["model_error_percentage"] = round(
                (data["error_model"] / data["total"]) * 100, 2)
            data["anomaly_percentage"] = round(
                (data["anomalies"] / data["total"]) * 100, 2)
            data["model_error_anomaly_percentage"] = round(
                (data["error_anomalies"] / data["error_model"]) * 100, 2) if data["error_model"] > 0 else 0.0
        else:
            data["anomaly_percentage"] = 0.0
            data["model_error_anomaly_percentage"] = 0.0

        # Coletar dados para calcular as estatísticas
        error_counts.append(data["model_error_percentage"])
        anomaly_counts.append(data["anomaly_percentage"])
        error_anomaly_counts.append(data["model_error_anomaly_percentage"])

        # Salvar a classe correspondente para valores mínimo e máximo
        error_classes[data["model_error_percentage"]] = class_name
        anomaly_classes[data["anomaly_percentage"]] = class_name
        error_anomaly_classes[data["model_error_anomaly_percentage"]] = class_name

    # Salva os relatórios
    with open(os.path.join(output_dir, f"{anomaly_method}_anomaly_classification_report.json"), 'w') as json_file:
        json.dump(anomaly_report, json_file, indent=4)

    # Calcula estatísticas gerais
    statistics_data = {
        "errors": {
            "mean": round(statistics.mean(error_counts), 2) if error_counts else 0.0,
            "min": {
                "value": min(error_counts) if error_counts else 0,
                "class": error_classes[min(error_counts)] if error_counts else ""
            },
            "max": {
                "value": max(error_counts) if error_counts else 0,
                "class": error_classes[max(error_counts)] if error_counts else ""
            },
            "mode": statistics.mode(error_counts) if len(error_counts) > 1 else error_counts[0] if error_counts else 0
        },
        "anomalies": {
            "mean": round(statistics.mean(anomaly_counts), 2) if anomaly_counts else 0.0,
            "min": {
                "value": min(anomaly_counts) if anomaly_counts else 0,
                "class": anomaly_classes[min(anomaly_counts)] if anomaly_counts else ""
            },
            "max": {
                "value": max(anomaly_counts) if anomaly_counts else 0,
                "class": anomaly_classes[max(anomaly_counts)] if anomaly_counts else ""
            },
            "mode": statistics.mode(anomaly_counts) if len(anomaly_counts) > 1 else anomaly_counts[0] if anomaly_counts else 0
        },
        "error_anomalies": {
            "mean": round(statistics.mean(error_anomaly_counts), 2) if error_anomaly_counts else 0.0,
            "min": {
                "value": min(error_anomaly_counts) if error_anomaly_counts else 0,
                "class": error_anomaly_classes[min(error_anomaly_counts)] if error_anomaly_counts else ""
            },
            "max": {
                "value": max(error_anomaly_counts) if error_anomaly_counts else 0,
                "class": error_anomaly_classes[max(error_anomaly_counts)] if error_anomaly_counts else ""
            },
            "mode": statistics.mode(error_anomaly_counts) if len(error_anomaly_counts) > 1 else error_anomaly_counts[0] if error_anomaly_counts else 0
        }
    }

    # Salva as estatísticas gerais em um arquivo JSON
    with open(os.path.join(output_dir, f"{anomaly_method}_anomaly_statistics.json"), 'w') as json_file:
        json.dump(statistics_data, json_file, indent=4)
