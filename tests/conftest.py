"""Shared fixtures.

The suite deliberately runs on a tiny synthetic problem: two qubits, a few
dozen samples and a handful of optimiser iterations. That keeps the whole run
in the low seconds while still exercising the real circuits, the real backends
and the real training loop. Only ``test_data.py`` touches the actual Breast
Cancer dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from qml_compare.config import CircuitConfig, ExperimentConfig, NoiseConfig, TrainingConfig
from qml_compare.data import Dataset


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def toy_dataset() -> Dataset:
    """Two well-separated blobs in 2-D, scaled to [-1, 1]."""
    generator = np.random.default_rng(7)
    n_per_class = 30
    class_zero = generator.normal(loc=[-0.5, -0.5], scale=0.18, size=(n_per_class, 2))
    class_one = generator.normal(loc=[0.5, 0.5], scale=0.18, size=(n_per_class, 2))
    x = np.clip(np.vstack([class_zero, class_one]), -1.0, 1.0)
    y = np.array([0] * n_per_class + [1] * n_per_class)

    order = generator.permutation(len(y))
    x, y = x[order], y[order]
    split = int(0.75 * len(y))

    return Dataset(
        x_train=x[:split],
        x_test=x[split:],
        y_train=y[:split],
        y_test=y[split:],
        class_names=("negative", "positive"),
        explained_variance_ratio=np.array([0.6, 0.4]),
        preprocessor=Pipeline([("scale", MinMaxScaler(feature_range=(-1.0, 1.0)))]),
    )


@pytest.fixture()
def fast_config() -> ExperimentConfig:
    """An ``ExperimentConfig`` small enough to run inside a unit test."""
    return ExperimentConfig(
        seed=3,
        depths=(1, 2),
        reference_depth=1,
        noise_sweep=(0.0, 0.05),
        noise_sweep_shots=(0,),
        circuit=CircuitConfig(feature_map_reps=1, depth=1),
        training=TrainingConfig(maxiter=12),
        noise=NoiseConfig(one_qubit_error=0.05, two_qubit_error=0.05, shots=0),
    )
