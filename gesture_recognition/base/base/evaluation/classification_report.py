from sklearn.metrics import classification_report, confusion_matrix
from base.base.utilities import csv_utils, folder_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


def report(results_path: str, y_true: list, y_pred: list,
           average_inference_time: float = None, throughput: float = None,
           save_results: bool = True):
    report = classification_report(y_true,
                                   y_pred,
                                   output_dict=True)
    # Cria um dicionário com as métricas agregadas
    report_dict = {
        "accuracy": [report["accuracy"]],
        "recall_macro_avg": [report["macro avg"]["recall"]],
        "precision_macro_avg": [report["macro avg"]["precision"]],
        "f1_macro_avg": [report["macro avg"]["f1-score"]],
        "recall_weighted": [report["weighted avg"]["recall"]],
        "precision_weighted": [report["weighted avg"]["precision"]],
        "f1_weighted": [report["weighted avg"]["f1-score"]],
        'average_inference_time': [average_inference_time],
        'throughput': [throughput]
    }

    # Converte o dicionário em um DataFrame com uma única linha
    df_report = pd.DataFrame(report_dict)

    # Gera DataFrames para cada classe
    dfs = []
    for label, metrics in report.items():
        if label not in ("accuracy", "macro avg", "weighted avg"):
            df_temp = pd.DataFrame(metrics, index=[label])
            dfs.append(df_temp)

    # Concatena os DataFrames de classes em um único DataFrame
    df_classes = pd.concat(dfs)
    df_classes.reset_index(inplace=True)
    df_classes.rename(columns={"index": "class"}, inplace=True)

    # Salva os resultados em CSVs, se necessário
    if save_results:
        folder_utils.create_directories_for_file(results_path)

        csv_utils.add_new_line(
            df_classes, f"{results_path}/report_per_classes.csv")
        csv_utils.add_new_line(df_report, f"{results_path}/report.csv")

    return df_classes, df_report


def confusion_matrix_report(results_path: str, y_true, y_pred):
    folder_utils.create_directories_for_file(
        f'{results_path}/confusion_matrix.png')
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(range(len(conf_matrix)), range(len(conf_matrix)))
    plt.yticks(range(len(conf_matrix)), range(len(conf_matrix)))
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig(f'{results_path}/confusion_matrix.png', bbox_inches='tight')


def normalized_confusion_matrix(results_path: str, y_true, y_pred, class_names):
    folder_utils.create_directories_for_file(
        f'{results_path}/confusion_matrix_normalized.png')

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))

    color_map = plt.cm.Blues
    normalized_conf_matrix = np.where(conf_matrix == 0, 0.5, conf_matrix)

    normed_cmap = np.log1p(normalized_conf_matrix)
    plt.imshow(normed_cmap, interpolation='nearest', cmap=color_map)

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if normalized_conf_matrix[i, j] > 500 else "black")

    plt.xticks(np.arange(len(class_names)),
               class_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Custom Scale Confusion Matrix')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        f'{results_path}/confusion_matrix_normalized.png', bbox_inches='tight')
