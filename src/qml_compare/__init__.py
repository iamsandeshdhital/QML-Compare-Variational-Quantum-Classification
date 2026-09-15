"""QML-Compare: variational quantum classification benchmarked against classical models.

Authors: Anurag Jha, Sandesh Dhital.

The package is organised as a small pipeline:

``config``      immutable experiment configuration loaded from JSON or the CLI
``data``        Breast Cancer loading, PCA reduction, scaling and splitting
``circuits``    ZZFeatureMap encoder + RealAmplitudes ansatz construction
``backends``    exact statevector and Aer (optionally noisy) circuit evaluation
``vqc``         the variational quantum classifier itself (scikit-learn style)
``classical``   Logistic Regression and SVM baselines
``experiments`` the three studies: baseline, circuit depth, hardware noise
``reporting``   JSON/CSV/Markdown artefacts
``plotting``    figures written to the results directory
"""

from qml_compare.config import ExperimentConfig
from qml_compare.data import Dataset, load_dataset
from qml_compare.vqc import VariationalQuantumClassifier

__all__ = [
    "ExperimentConfig",
    "Dataset",
    "load_dataset",
    "VariationalQuantumClassifier",
    "__version__",
    "__authors__",
]

__version__ = "1.0.0"
__authors__ = ("Anurag Jha", "Sandesh Dhital")
