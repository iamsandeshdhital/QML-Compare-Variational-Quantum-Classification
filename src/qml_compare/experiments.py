"""The four studies that make up QML-Compare.

``baseline``     classical models vs. the VQC at the reference depth
``depth``        the VQC at every depth in ``config.depths``
``noise``        ideal vs. noisy inference vs. noise-aware training
``noise-sweep``  ideal-trained weights deployed across a range of error rates

They overlap: the reference-depth noiseless VQC appears in all four. Training
it once and reusing the fitted model keeps a full run honest (the same weights
back every table) and cuts most of the wall-clock cost.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator

from qml_compare.classical import build_all_classical_models
from qml_compare.config import ExperimentConfig
from qml_compare.data import Dataset, load_dataset
from qml_compare.logging_utils import get_logger
from qml_compare.metrics import ModelResult, classification_metrics
from qml_compare.vqc import VariationalQuantumClassifier

LOGGER = get_logger(__name__)


@dataclass
class ExperimentSuite:
    """Everything one run produced, ready to be serialised."""

    config: dict[str, Any]
    dataset: dict[str, Any]
    environment: dict[str, Any]
    results: list[ModelResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "dataset": self.dataset,
            "environment": self.environment,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentSuite":
        """Rebuild a suite from a ``results.json`` payload.

        Lets the reports and figures be regenerated from a finished run
        without touching a simulator again.
        """
        return cls(
            config=dict(payload.get("config", {})),
            dataset=dict(payload.get("dataset", {})),
            environment=dict(payload.get("environment", {})),
            results=[ModelResult(**record) for record in payload.get("results", [])],
            started_at=str(payload.get("started_at", "")),
            finished_at=str(payload.get("finished_at", "")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentSuite":
        """Load a suite from a ``results.json`` file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def by_experiment(self, name: str) -> list[ModelResult]:
        """Records belonging to one study, in insertion order."""
        return [result for result in self.results if result.experiment == name]

    def find(self, model: str, experiment: str | None = None) -> ModelResult | None:
        """First record matching a model name (optionally within one study)."""
        for result in self.results:
            if result.model == model and (experiment is None or result.experiment == experiment):
                return result
        return None


# --------------------------------------------------------------- evaluation --
def evaluate_classical(
    name: str,
    model: BaseEstimator,
    dataset: Dataset,
    experiment: str,
) -> ModelResult:
    """Fit and score one scikit-learn baseline."""
    start = time.perf_counter()
    model.fit(dataset.x_train, dataset.y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(dataset.x_test)
    inference_time = time.perf_counter() - start

    scores = _positive_class_scores(model, dataset.x_test)
    metrics = classification_metrics(dataset.y_test, y_pred, scores)
    train_accuracy = float(np.mean(model.predict(dataset.x_train) == dataset.y_train))

    LOGGER.info("%-22s test accuracy %.4f (%.3fs)", name, metrics["accuracy"], training_time)
    return ModelResult(
        model=name,
        family="classical",
        experiment=experiment,
        train_accuracy=train_accuracy,
        test_accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        training_time=training_time,
        inference_time=inference_time,
        confusion_matrix=metrics["confusion_matrix"],
        details={"estimator": type(model).__name__, "params": _jsonable_params(model)},
    )


def evaluate_vqc(
    name: str,
    classifier: VariationalQuantumClassifier,
    dataset: Dataset,
    experiment: str,
) -> ModelResult:
    """Score an already-fitted VQC on the held-out split."""
    start = time.perf_counter()
    y_pred = classifier.predict(dataset.x_test)
    inference_time = time.perf_counter() - start

    scores = classifier.predict_proba(dataset.x_test)[:, 1]
    metrics = classification_metrics(dataset.y_test, y_pred, scores)
    train_accuracy = classifier.score(dataset.x_train, dataset.y_train)

    LOGGER.info(
        "%-22s test accuracy %.4f (%.2fs training)",
        name,
        metrics["accuracy"],
        classifier.history.training_time,
    )
    return ModelResult(
        model=name,
        family="quantum",
        experiment=experiment,
        train_accuracy=train_accuracy,
        test_accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        training_time=classifier.history.training_time,
        inference_time=inference_time,
        confusion_matrix=metrics["confusion_matrix"],
        details=classifier.describe(),
        history=classifier.history.to_dict(),
    )


def train_vqc(
    config: ExperimentConfig,
    dataset: Dataset,
    depth: int,
    noisy: bool = False,
) -> VariationalQuantumClassifier:
    """Fit one VQC at ``depth``, on the ideal or the noisy backend."""
    noise = replace(config.noise, enabled=noisy)
    LOGGER.info(
        "Training VQC (depth=%d, %s)...", depth, "noisy" if noisy else "ideal statevector"
    )
    classifier = VariationalQuantumClassifier(
        depth=depth,
        circuit_config=config.circuit,
        training_config=config.training,
        noise_config=noise,
        seed=config.seed,
    )
    classifier.fit(dataset.x_train, dataset.y_train)
    return classifier


# ------------------------------------------------------------------ studies --
class _VQCCache:
    """Memoises fitted VQCs so overlapping studies train each model once."""

    def __init__(self, config: ExperimentConfig, dataset: Dataset) -> None:
        self._config = config
        self._dataset = dataset
        self._store: dict[tuple[int, bool], VariationalQuantumClassifier] = {}

    def get(self, depth: int, noisy: bool = False) -> VariationalQuantumClassifier:
        key = (int(depth), bool(noisy))
        if key not in self._store:
            self._store[key] = train_vqc(self._config, self._dataset, depth, noisy)
        else:
            LOGGER.info("Reusing cached VQC (depth=%d, noisy=%s)", depth, noisy)
        return self._store[key]


def run_baseline_study(
    config: ExperimentConfig,
    dataset: Dataset,
    cache: _VQCCache | None = None,
) -> list[ModelResult]:
    """Classical models vs. the VQC at the reference depth, all noiseless."""
    LOGGER.info("=== Study 1: classical baselines vs. VQC ===")
    cache = cache or _VQCCache(config, dataset)
    results = [
        evaluate_classical(name, model, dataset, "baseline")
        for name, model in build_all_classical_models(config.seed).items()
    ]
    depth = config.reference_depth
    classifier = cache.get(depth, noisy=False)
    results.append(evaluate_vqc(f"VQC (Depth {depth})", classifier, dataset, "baseline"))
    return results


def run_depth_study(
    config: ExperimentConfig,
    dataset: Dataset,
    cache: _VQCCache | None = None,
) -> list[ModelResult]:
    """Accuracy and training cost as the ansatz gains repetitions."""
    LOGGER.info("=== Study 2: circuit depth sweep %s ===", list(config.depths))
    cache = cache or _VQCCache(config, dataset)
    results = []
    for depth in config.depths:
        classifier = cache.get(depth, noisy=False)
        results.append(evaluate_vqc(f"VQC Depth {depth}", classifier, dataset, "depth"))
    return results


def run_noise_study(
    config: ExperimentConfig,
    dataset: Dataset,
    cache: _VQCCache | None = None,
) -> list[ModelResult]:
    """Three ways noise can reach a VQC, separated.

    Reporting a single "with noise" number conflates two very different
    situations, so the study splits them:

    * **Ideal** -- the noiseless reference.
    * **Noisy inference** -- weights trained in simulation, then deployed on a
      noisy device. This is what "running your model on hardware" means.
    * **Noise-aware training** -- the optimiser itself sees the noisy device
      and can compensate for it.
    """
    depth = config.reference_depth
    rate = config.noise.one_qubit_error
    LOGGER.info("=== Study 3: noise sensitivity at depth %d ===", depth)
    cache = cache or _VQCCache(config, dataset)

    ideal = cache.get(depth, noisy=False)
    deployed = ideal.transfer_to(replace(config.noise, enabled=True), seed=config.seed)
    retrained = cache.get(depth, noisy=True)

    return [
        evaluate_vqc("Ideal Simulator (no noise)", ideal, dataset, "noise"),
        evaluate_vqc(f"Noisy Inference ({rate:.1%} depolarizing)", deployed, dataset, "noise"),
        evaluate_vqc(f"Noise-Aware Training ({rate:.1%} depolarizing)", retrained, dataset, "noise"),
    ]


def run_noise_sweep(
    config: ExperimentConfig,
    dataset: Dataset,
    cache: _VQCCache | None = None,
) -> list[ModelResult]:
    """Deploy the ideal-trained weights across a range of gate error rates.

    Inference only -- no retraining -- so the sweep is cheap and isolates how
    fast the read-out signal decays as the device gets worse. Each error rate
    is evaluated both exactly and at a finite shot budget, because the two
    effects compound: depolarising noise shrinks the margin around the 0.5
    decision threshold, and sampling noise then flips whatever is left near it.
    """
    depth = config.reference_depth
    LOGGER.info("=== Study 4: noise sweep %s ===", list(config.noise_sweep))
    cache = cache or _VQCCache(config, dataset)
    ideal = cache.get(depth, noisy=False)

    results: list[ModelResult] = []
    for rate in config.noise_sweep:
        for shots in config.noise_sweep_shots:
            if rate == 0.0 and shots == 0:
                model, label = ideal, "Ideal (0.0%, exact)"
            else:
                noise = replace(
                    config.noise,
                    enabled=True,
                    one_qubit_error=rate,
                    two_qubit_error=rate,
                    shots=shots,
                )
                readout = "exact" if shots == 0 else f"{shots} shots"
                label = f"Depolarizing {rate:.1%} ({readout})"
                model = ideal.transfer_to(noise, seed=config.seed)

            result = evaluate_vqc(label, model, dataset, "noise_sweep")
            result.details["error_rate"] = float(rate)
            result.details["sweep_shots"] = int(shots)
            result.details["mean_margin"] = _mean_margin(model, dataset.x_test)
            results.append(result)
    return results


STUDIES = {
    "baseline": run_baseline_study,
    "depth": run_depth_study,
    "noise": run_noise_study,
    "noise-sweep": run_noise_sweep,
}


def run_suite(
    config: ExperimentConfig,
    studies: tuple[str, ...] = ("baseline", "depth", "noise", "noise-sweep"),
    dataset: Dataset | None = None,
) -> ExperimentSuite:
    """Run the requested studies and collect them into one suite."""
    unknown = sorted(set(studies) - set(STUDIES))
    if unknown:
        raise ValueError(f"unknown study/studies: {', '.join(unknown)}")

    started = datetime.now(timezone.utc)
    dataset = dataset or load_dataset(config.data, seed=config.seed)
    cache = _VQCCache(config, dataset)

    results: list[ModelResult] = []
    for study in studies:
        results.extend(STUDIES[study](config, dataset, cache))

    suite = ExperimentSuite(
        config=config.to_dict(),
        dataset=dataset.summary(),
        environment=environment_info(),
        results=results,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    LOGGER.info("Completed %d model evaluations", len(results))
    return suite


# ------------------------------------------------------------------ helpers --
def environment_info() -> dict[str, Any]:
    """Versions and machine details, so a results file can be traced back."""
    import qiskit
    import scipy
    import sklearn

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
    }
    try:
        import qiskit_aer

        info["qiskit_aer"] = qiskit_aer.__version__
    except ImportError:  # pragma: no cover - Aer is a hard dependency in practice
        info["qiskit_aer"] = "not installed"
    return info


def _mean_margin(classifier: VariationalQuantumClassifier, x: np.ndarray) -> float:
    """Average distance of ``P(y=1)`` from the 0.5 decision threshold.

    Accuracy can hold up long after the read-out signal has collapsed, so the
    margin is the metric that actually shows noise eating the model.
    """
    return float(np.mean(np.abs(classifier.predict_proba(x)[:, 1] - 0.5)))


def _positive_class_scores(model: BaseEstimator, x: np.ndarray) -> np.ndarray:
    """Continuous class-1 score from whichever API the estimator exposes."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x))
    return np.asarray(model.predict(x), dtype=float)


def _jsonable_params(model: BaseEstimator) -> dict[str, Any]:
    """Estimator hyper-parameters reduced to JSON-safe scalars."""
    params = {}
    for key, value in model.get_params().items():
        params[key] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
    return params
