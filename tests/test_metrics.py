"""Metric computation and the result record."""

from __future__ import annotations

import numpy as np
import pytest

from qml_compare.metrics import (
    ModelResult,
    accuracy_drop,
    classification_metrics,
    relative_change,
)


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 0, 1, 1])
    metrics = classification_metrics(y, y, y.astype(float))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 3]]


def test_confusion_matrix_orientation():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = classification_metrics(y_true, y_pred)["confusion_matrix"]
    # rows are truth, columns are prediction
    assert matrix == [[1, 1], [0, 2]]


def test_roc_auc_uses_the_continuous_score():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1])  # one mistake
    scores = np.array([0.1, 0.2, 0.49, 0.9])  # but the ranking is perfect
    metrics = classification_metrics(y_true, y_pred, scores)
    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] == 1.0


def test_single_class_split_gives_nan_auc_not_an_error():
    y = np.zeros(4, dtype=int)
    metrics = classification_metrics(y, y, np.linspace(0, 1, 4))
    assert np.isnan(metrics["roc_auc"])


def test_accuracy_drop_is_in_percentage_points():
    assert accuracy_drop(0.947, 0.782) == pytest.approx(16.5, abs=1e-6)
    assert accuracy_drop(0.9, 0.95) == pytest.approx(-5.0, abs=1e-6)


def test_relative_change():
    assert relative_change(10.0, 15.0) == pytest.approx(50.0)
    assert np.isnan(relative_change(0.0, 1.0))


def test_model_result_row_is_flat_and_rounded():
    result = ModelResult(
        model="VQC",
        family="quantum",
        experiment="baseline",
        train_accuracy=0.912345,
        test_accuracy=0.887654,
        precision=0.9,
        recall=0.8,
        f1=0.847,
        roc_auc=0.95,
        training_time=12.3456,
        inference_time=0.1234,
        confusion_matrix=[[10, 2], [1, 20]],
    )
    row = result.row()
    assert row["test_accuracy"] == 0.8877
    assert row["training_time_s"] == 12.346
    assert not any(isinstance(value, (list, dict)) for value in row.values())
    assert "confusion_matrix" in result.to_dict()
