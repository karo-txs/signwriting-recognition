from __future__ import annotations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Sequence, Dict, Any
from pathlib import Path
import json, logging
import pandas as pd
import numpy as np


def _bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int,
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for i in range(n_iterations):
        idx = rng.integers(0, len(y_true), sample_size)
        yt, yp = y_true[idx], y_pred[idx]
        rows.append(
            dict(
                iteration=i + 1,
                accuracy=accuracy_score(yt, yp),
                precision=precision_score(yt, yp, average="weighted", zero_division=0),
                recall=recall_score(yt, yp, average="weighted", zero_division=0),
                f1_score=f1_score(yt, yp, average="weighted", zero_division=0),
            )
        )
    return pd.DataFrame(rows)


def _ci_bounds(series: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo (1-alpha) usando percentis bootstrap."""
    lower = np.percentile(series, 100 * alpha / 2)
    upper = np.percentile(series, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def _stat_dict(series: pd.Series) -> dict[str, float]:
    mean = float(series.mean())
    low, high = _ci_bounds(series)
    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "pm": f"{mean:.4f} ± {(high - low) / 2:.5f}",
    }


def save_bootstrap_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    report_dir: Path,
    n_iterations: int = 1_000,
    sample_frac: float = 0.4,
    rng_seed: int | None = None,
) -> Dict[str, Any]:
    """
    Executa bootstrap sobre (y_true, y_pred) **sem** repetir inferência.

    Retorna um dicionário resumo e grava arquivos em `report_dir`.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_total = len(y_true_arr)
    sample_size = int(sample_frac * n_total)

    rng = np.random.default_rng(rng_seed)

    # iterações
    df_iter = _bootstrap_metrics(y_true_arr, y_pred_arr, n_iterations, sample_size, rng)
    df_iter.to_csv(report_dir / "bootstrap_iterations.csv", index=False)

    # resumo
    def _stat(col: str):
        return dict(
            mean=float(df_iter[col].mean()),
            min=float(df_iter[col].min()),
            max=float(df_iter[col].max()),
        )

    summary = dict(
        n_iterations=n_iterations,
        sample_size=sample_size,
        accuracy=_stat_dict(df_iter["accuracy"]),
        precision=_stat_dict(df_iter["precision"]),
        recall=_stat_dict(df_iter["recall"]),
        f1_score=_stat_dict(df_iter["f1_score"]),
    )

    (report_dir / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )
    logging.info("Bootstrap salvo em %s", report_dir)
    return summary
