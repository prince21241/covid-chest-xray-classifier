"""Threshold selection and binary metrics. COVID is the positive class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD_RULE = (
    "Choose the threshold that maximizes COVID F1 on the validation set. "
    "Ties break toward a higher Youden index (sensitivity + specificity - 1), "
    "then toward 0.5. The test set is not used to choose or break the tie."
)


def _curve_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    try:
        roc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc = None
    try:
        pr = float(average_precision_score(labels, scores))
    except ValueError:
        pr = None
    return roc, pr


def binary_metrics(labels, scores, threshold: float) -> dict:
    """Score a frozen threshold. ``scores`` are model scores, not hard labels."""
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    roc_auc, pr_auc = _curve_scores(y_true, y_score)
    return {
        "n": int(y_true.size),
        "n_covid": int((y_true == 1).sum()),
        "n_normal": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["Normal", "COVID"],
        "confusion_matrix_axes": "Rows are true labels and columns are predicted labels, ordered Normal then COVID.",
    }


def select_threshold(labels, scores) -> tuple[float, dict]:
    """Pick a decision threshold from validation labels and scores only."""
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    candidates = np.round(np.linspace(0.01, 0.99, 99), 2)
    best_key = None
    best_threshold = 0.5
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_score, float(threshold))
        youden = metrics["sensitivity"] + metrics["specificity"] - 1.0
        key = (metrics["f1"], youden, -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, binary_metrics(y_true, y_score, best_threshold)


def always_normal_baseline(labels) -> dict:
    """Metrics for a classifier that predicts Normal for every image.

    Ranking metrics are defined directly: a constant score has ROC-AUC 0.5, and
    its precision-recall AUC equals the COVID prevalence.
    """
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.zeros_like(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    n_covid = int((y_true == 1).sum())
    prevalence = float(n_covid / y_true.size) if y_true.size else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "rule": "Predict Normal for every image.",
        "n": int(y_true.size),
        "n_covid": n_covid,
        "n_normal": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": 0.5,
        "pr_auc": prevalence,
        "roc_auc_note": "A constant score has no ranking ability, so ROC-AUC is 0.5.",
        "pr_auc_note": "PR-AUC of an uninformative classifier equals the COVID prevalence.",
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["Normal", "COVID"],
    }
