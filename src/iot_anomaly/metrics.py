"""Detection metrics used by the SKAB benchmark plus threshold-free ROC-AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Point-wise precision/recall/F1 and SKAB's false/missing alarm rates (in %)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": 100.0 * fp / (fp + tn) if fp + tn else 0.0,
        "mar": 100.0 * fn / (fn + tp) if fn + tp else 0.0,
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))
