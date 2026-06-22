"""Evaluation utilities for binary classification.

All functions are pure (no side effects) and tested without TensorFlow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a standard set of binary-classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0 = real, 1 = fake).
    y_pred:
        Predicted labels.
    y_proba:
        Predicted probability of the *positive* class (class 1 = fake).
        When provided, ``roc_auc`` is computed; otherwise it is ``None``.

    Returns
    -------
    dict[str, Any]
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc`` (float or None), ``confusion_matrix`` (2-D array).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def format_report(name: str, metrics: dict[str, Any]) -> str:
    """Return a human-readable one-block report for a single model.

    Parameters
    ----------
    name:
        Model display name (e.g. ``"Logistic Regression"``).
    metrics:
        Output of :func:`evaluate`.

    Returns
    -------
    str
        Formatted multi-line string.
    """
    roc = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "n/a"
    cm = metrics["confusion_matrix"]
    lines = [
        f"=== {name} ===",
        f"  Accuracy : {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall   : {metrics['recall']:.4f}",
        f"  F1       : {metrics['f1']:.4f}",
        f"  ROC-AUC  : {roc}",
        "  Confusion matrix (rows=actual, cols=predicted):",
        f"    TN={cm[0,0]}  FP={cm[0,1]}",
        f"    FN={cm[1,0]}  TP={cm[1,1]}",
    ]
    return "\n".join(lines)
