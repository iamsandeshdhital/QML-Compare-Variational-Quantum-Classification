"""Classical baselines.

The point of the comparison only holds if the baselines see exactly the same
data as the VQC: the same PCA projection, the same ``[-1, 1]`` scaling and the
same split. So they take a prepared :class:`~qml_compare.data.Dataset` rather
than loading anything themselves.
"""

from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#: Baseline name -> factory. Keyed by the label used in every report.
CLASSICAL_MODELS: dict[str, Callable[[int], BaseEstimator]] = {
    "Logistic Regression": lambda seed: LogisticRegression(
        max_iter=1000,
        random_state=seed,
    ),
    # No ``probability=True``: it fits an extra internal cross-validated
    # calibration and is deprecated in recent scikit-learn. ROC-AUC only needs
    # a ranking, and ``decision_function`` provides one for free.
    "SVM (RBF Kernel)": lambda seed: SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=seed,
    ),
}


def build_classical_model(name: str, seed: int = 42) -> BaseEstimator:
    """Instantiate one baseline by name."""
    try:
        factory = CLASSICAL_MODELS[name]
    except KeyError:
        known = ", ".join(sorted(CLASSICAL_MODELS))
        raise KeyError(f"unknown classical model {name!r}; expected one of: {known}") from None
    return factory(seed)


def build_all_classical_models(seed: int = 42) -> dict[str, BaseEstimator]:
    """Instantiate every baseline, in report order."""
    return {name: build_classical_model(name, seed) for name in CLASSICAL_MODELS}
