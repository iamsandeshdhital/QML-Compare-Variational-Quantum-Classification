"""Dataset preparation: shapes, ranges, determinism and leakage."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from qml_compare.config import DataConfig
from qml_compare.data import build_preprocessor, load_dataset


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DataConfig(), seed=42)


def test_shapes_and_split(dataset):
    assert dataset.n_features == 4
    assert dataset.n_train + dataset.n_test == 569  # Breast Cancer Wisconsin
    assert dataset.n_test == pytest.approx(0.2 * 569, abs=1)
    assert dataset.x_train.shape == (dataset.n_train, 4)
    assert dataset.x_test.shape == (dataset.n_test, 4)


def test_features_are_within_the_rotation_range(dataset):
    for split in (dataset.x_train, dataset.x_test):
        assert split.min() >= -1.0
        assert split.max() <= 1.0
    # The training split is what the scaler was fitted on, so it should reach
    # both ends of the interval.
    assert dataset.x_train.min() == pytest.approx(-1.0)
    assert dataset.x_train.max() == pytest.approx(1.0)


def test_labels_are_binary_and_stratified(dataset):
    assert set(np.unique(dataset.y_train)) == {0, 1}
    train_ratio = dataset.y_train.mean()
    test_ratio = dataset.y_test.mean()
    assert train_ratio == pytest.approx(test_ratio, abs=0.02)


def test_pca_keeps_a_useful_share_of_the_variance(dataset):
    assert 0.6 < dataset.total_explained_variance < 1.0
    ratios = dataset.explained_variance_ratio
    assert np.all(np.diff(ratios) <= 0), "components should be ordered by variance"


def test_loading_is_deterministic_for_a_seed():
    first = load_dataset(DataConfig(), seed=1)
    second = load_dataset(DataConfig(), seed=1)
    np.testing.assert_allclose(first.x_train, second.x_train)

    different = load_dataset(DataConfig(), seed=2)
    assert not np.allclose(first.x_train, different.x_train)


def test_preprocessor_is_fitted_on_training_data_only(dataset):
    """The test split must not influence the PCA basis or the scaler.

    Rebuilding the same split by hand and fitting a fresh pipeline on the
    training rows alone has to reproduce the stored transform exactly. If any
    test-set statistic had leaked into the fit, it would not.
    """
    raw = load_breast_cancer()
    x_train_raw, x_test_raw, _, _ = train_test_split(
        raw.data, raw.target, test_size=0.2, random_state=42, stratify=raw.target
    )

    clean = build_preprocessor(DataConfig())
    clean.set_params(pca__random_state=42)
    np.testing.assert_allclose(clean.fit_transform(x_train_raw), dataset.x_train, atol=1e-12)
    np.testing.assert_allclose(
        np.clip(clean.transform(x_test_raw), -1.0, 1.0), dataset.x_test, atol=1e-12
    )


def test_component_count_is_configurable():
    dataset = load_dataset(DataConfig(n_components=2), seed=42)
    assert dataset.n_features == 2
    assert dataset.x_train.shape[1] == 2


def test_build_preprocessor_can_skip_standardisation():
    pipeline = build_preprocessor(DataConfig(standardize_before_pca=False))
    assert "standardize" not in pipeline.named_steps
    assert list(pipeline.named_steps) == ["pca", "scale"]


def test_summary_is_serialisable(dataset):
    summary = dataset.summary()
    assert summary["n_features"] == 4
    assert set(summary["class_balance_train"]) == {"0", "1"}
    assert isinstance(summary["total_explained_variance"], float)
