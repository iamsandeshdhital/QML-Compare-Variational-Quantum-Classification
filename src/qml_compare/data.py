"""Dataset loading and preprocessing.

The Breast Cancer Wisconsin dataset has 30 features, which is far more than a
laptop-scale quantum simulation can absorb: one qubit per feature would mean a
2**30 amplitude statevector. We therefore compress it to ``n_components``
principal components (4 by default) and rescale to ``[-1, 1]`` so the values
can be used directly as rotation angles.

Every transformer is fitted on the training split only and then applied to the
test split. Fitting PCA or the scaler on the full dataset before splitting
leaks test-set statistics into training and inflates the reported accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qml_compare.config import DataConfig
from qml_compare.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class Dataset:
    """A fully preprocessed train/test split ready for both model families."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]
    explained_variance_ratio: np.ndarray
    preprocessor: Pipeline

    @property
    def n_features(self) -> int:
        """Number of features after dimensionality reduction (== number of qubits)."""
        return int(self.x_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.x_test.shape[0])

    @property
    def total_explained_variance(self) -> float:
        """Fraction of the original variance retained by the PCA projection."""
        return float(np.sum(self.explained_variance_ratio))

    def summary(self) -> dict[str, object]:
        """Small dictionary embedded into the results report."""
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "class_names": list(self.class_names),
            "class_balance_train": _class_balance(self.y_train),
            "class_balance_test": _class_balance(self.y_test),
            "explained_variance_ratio": [round(v, 6) for v in self.explained_variance_ratio],
            "total_explained_variance": round(self.total_explained_variance, 6),
            "feature_min": float(np.min(self.x_train)),
            "feature_max": float(np.max(self.x_train)),
        }


def build_preprocessor(config: DataConfig) -> Pipeline:
    """Assemble the standardise -> PCA -> min-max pipeline described in the docs."""
    steps: list[tuple[str, object]] = []
    if config.standardize_before_pca:
        # PCA maximises variance, so without standardisation the components are
        # dominated by whichever raw feature happens to have the largest units.
        steps.append(("standardize", StandardScaler()))
    steps.append(("pca", PCA(n_components=config.n_components)))
    steps.append(("scale", MinMaxScaler(feature_range=config.feature_range)))
    return Pipeline(steps)


def load_dataset(config: DataConfig | None = None, seed: int = 42) -> Dataset:
    """Load and preprocess the Breast Cancer Wisconsin dataset.

    Args:
        config: preprocessing settings; defaults are 4 components and a 20% test split.
        seed: controls the stratified split and the PCA solver.

    Returns:
        A :class:`Dataset` whose features live in the configured range.
    """
    config = config or DataConfig()
    raw = load_breast_cancer()
    x_raw, y = raw.data, raw.target

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=config.test_size,
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(config)
    preprocessor.set_params(pca__random_state=seed)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)

    # MinMaxScaler is fitted on training data only, so unseen test points may
    # fall marginally outside the range. Rotation angles stay well defined, but
    # we clip to keep the encoding strictly inside the documented interval.
    low, high = config.feature_range
    x_test = np.clip(x_test, low, high)

    dataset = Dataset(
        x_train=np.ascontiguousarray(x_train, dtype=float),
        x_test=np.ascontiguousarray(x_test, dtype=float),
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        class_names=tuple(str(name) for name in raw.target_names),
        explained_variance_ratio=preprocessor.named_steps["pca"].explained_variance_ratio_,
        preprocessor=preprocessor,
    )
    LOGGER.info(
        "Dataset ready: %d train / %d test samples, %d features, %.1f%% variance retained",
        dataset.n_train,
        dataset.n_test,
        dataset.n_features,
        100 * dataset.total_explained_variance,
    )
    return dataset


def _class_balance(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}
