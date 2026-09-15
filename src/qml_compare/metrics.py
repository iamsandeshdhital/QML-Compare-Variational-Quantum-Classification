"""Evaluation metrics and the record type every experiment produces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass
class ModelResult:
    """One trained model evaluated on one dataset split.

    ``experiment`` groups records ("baseline", "depth", "noise") so a single
    flat results file can back every table and figure.
    """

    model: str
    family: str
    experiment: str
    train_accuracy: float
    test_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    training_time: float
    inference_time: float
    confusion_matrix: list[list[int]]
    details: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def row(self) -> dict[str, Any]:
        """Flat view used for the CSV and Markdown tables."""
        return {
            "experiment": self.experiment,
            "model": self.model,
            "family": self.family,
            "test_accuracy": round(self.test_accuracy, 4),
            "train_accuracy": round(self.train_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "training_time_s": round(self.training_time, 3),
            "inference_time_s": round(self.inference_time, 3),
        }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    ``y_score`` is the continuous class-1 score used for ROC-AUC; when it is
    missing the hard predictions are used, which still gives a usable (if
    coarse) number.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    scores = y_pred if y_score is None else np.asarray(y_score).ravel()

    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        # Raised when the split happens to contain a single class.
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def accuracy_drop(reference: float, degraded: float) -> float:
    """Percentage points lost between two accuracies."""
    return float(100.0 * (reference - degraded))


def relative_change(reference: float, other: float) -> float:
    """Percentage change of ``other`` relative to ``reference``."""
    if reference == 0:
        return float("nan")
    return float(100.0 * (other - reference) / reference)
