from __future__ import annotations
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from core.eval.interfaces.eval_metrics import EvalMetrics
from typing import Sequence, Dict, Any
import matplotlib.pyplot as plt
import logging, itertools, json
from pathlib import Path
import pandas as pd
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_classification_csvs(
    out_dir: Path, metrics: EvalMetrics, labels: Sequence[str] | None = None
) -> None:
    _ensure_dir(out_dir)

    report_dict = classification_report(
        metrics.y_true,
        metrics.y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    df_header = pd.DataFrame(
        {
            "accuracy": [report_dict["accuracy"]],
            "recall_macro_avg": [report_dict["macro avg"]["recall"]],
            "precision_macro_avg": [report_dict["macro avg"]["precision"]],
            "f1_macro_avg": [report_dict["macro avg"]["f1-score"]],
            "recall_weighted": [report_dict["weighted avg"]["recall"]],
            "precision_weighted": [report_dict["weighted avg"]["precision"]],
            "f1_weighted": [report_dict["weighted avg"]["f1-score"]],
            "average_inference_time": [metrics.avg_time],
            "throughput": [metrics.throughput],
        }
    )
    df_header.to_csv(out_dir / "report.csv", index=False)

    class_rows = {
        k: v
        for k, v in report_dict.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    df_classes = (
        pd.DataFrame(class_rows).T.reset_index().rename(columns={"index": "class"})
    )
    df_classes.to_csv(out_dir / "report_per_classes.csv", index=False)


def _plot_confusion_matrices(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> None:
    _ensure_dir(out_dir)

    cm = confusion_matrix(y_true, y_pred)
    names = class_names or list(range(cm.shape[0]))

    def _plot(cm_mat, path: Path, title: str, norm: bool = False):
        plt.figure(figsize=(10, 10))
        plt.imshow(cm_mat, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(names))
        plt.xticks(tick_marks, names, rotation=45, ha="right")
        plt.yticks(tick_marks, names)

        thresh = cm_mat.max() / 2.0
        for i, j in itertools.product(range(cm_mat.shape[0]), range(cm_mat.shape[1])):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_mat[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    cm_norm = np.log1p(np.where(cm == 0, 0.5, cm))

    _plot(cm, out_dir / "confusion_matrix.png", "Confusion Matrix")
    _plot(
        cm_norm,
        out_dir / "confusion_matrix_normalized.png",
        "Confusion Matrix (log-scaled)",
        norm=True,
    )


def save_full_report(
    report_dir: Path,
    metrics_dict: Dict[str, Any],
    class_names: Sequence[str] | None = None,
) -> None:
    """
    Cria:
    <base_dir>/models/<model_name>/reports/classification_report/
        ├── report.csv
        ├── report_per_classes.csv
        ├── confusion_matrix.png
        └── confusion_matrix_normalized.png
    """
    y_true = metrics_dict["y_true"]
    y_pred = metrics_dict["y_pred"]

    metrics = EvalMetrics(
        y_true=y_true,
        y_pred=y_pred,
        avg_time=float(metrics_dict["average_inference_time"]),
        throughput=float(metrics_dict["throughput"]),
    )

    _save_classification_csvs(report_dir, metrics, labels=class_names)
    _plot_confusion_matrices(report_dir, y_true, y_pred, class_names)

    # opcional: JSON resumido
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                **metrics.scores(),
                "average_inference_time": metrics.avg_time,
                "throughput": metrics.throughput,
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    logging.info("Relatórios salvos em %s", report_dir)


def load_class_names(path: str | Path) -> List[str]:
    """
    Procura por **um único** arquivo que termine com `info.json`
    e devolve a lista de classes ordenada pelo id.

    Argumentos
    ----------
    path :  str | Path
        • diretório onde está o *_info.json
        • OU o próprio arquivo *.json
        • OU um pattern com glob (ex.: "/dir/*info.json")

    Retorna
    -------
    list[str]  — nomes das classes na ordem do id.
    """
    path = Path(path)

    if path.is_file():
        json_path = path

    elif "*" in path.name:
        matches = list(path.parent.glob(path.name))
        if len(matches) != 1:
            raise RuntimeError(
                f"Esperava 1 arquivo, encontrado(s) {len(matches)} para pattern {path}"
            )
        json_path = matches[0]

    elif path.is_dir():
        matches = list(path.glob("*info.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Esperava 1 '*info.json' em {path}, encontrado(s): {matches}"
            )
        json_path = matches[0]

    else:
        raise FileNotFoundError(f"Caminho inexistente: {path}")

    logging.info("Carregando classes de: %s", json_path)

    with open(json_path, encoding="utf-8") as f:
        name_to_id: Dict[str, int] = json.load(f)["labels"]

    id_name_pairs = sorted(name_to_id.items(), key=lambda kv: kv[1])
    return [name for name, _id in id_name_pairs]