"""The Variational Quantum Classifier.

Training is the usual hybrid loop:

1. encode a batch of samples with the ZZFeatureMap,
2. evolve them with the RealAmplitudes ansatz parameterised by ``theta``,
3. read out the parity of the measured bitstring to get ``P(y = 1 | x)``,
4. score that against the labels with binary cross-entropy,
5. hand the scalar loss to COBYLA, which proposes the next ``theta``.

Parity read-out is the Pauli-Z expectation value in disguise: an even-parity
bitstring contributes +1 and an odd-parity one -1, so the class probability is
``P(y = 1) = (1 - <Z on every qubit>) / 2``. Working with the probability
directly is what lets us use cross-entropy instead of a squared error, and it
costs nothing because both backends already return a full distribution.

COBYLA is gradient-free, which suits a noisy objective: as soon as the backend
reads out with a finite shot budget the loss surface becomes stochastic, and a
finite-difference gradient would mostly measure sampling noise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from qml_compare.backends import CircuitBackend, make_backend
from qml_compare.circuits import CircuitBundle, build_circuit
from qml_compare.config import CircuitConfig, NoiseConfig, TrainingConfig
from qml_compare.logging_utils import get_logger

LOGGER = get_logger(__name__)

_EPS = 1e-9


@dataclass
class TrainingHistory:
    """Everything worth plotting or auditing after a fit."""

    loss: list[float] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    n_iterations: int = 0
    training_time: float = 0.0
    converged: bool = False
    message: str = ""
    final_loss: float = float("nan")
    circuit_executions: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": [float(v) for v in self.loss],
            "elapsed": [float(v) for v in self.elapsed],
            "n_iterations": int(self.n_iterations),
            "training_time": float(self.training_time),
            "converged": bool(self.converged),
            "message": self.message,
            "final_loss": float(self.final_loss),
            "circuit_executions": int(self.circuit_executions),
        }


def parity_readout(num_qubits: int) -> np.ndarray:
    """Vector of class-1 weights, one per computational basis state.

    Entry ``i`` is 1 when the bit-string of ``i`` has odd parity and 0
    otherwise, so ``probabilities @ parity`` is ``P(y = 1)``.
    """
    indices = np.arange(2**num_qubits)
    parity = np.zeros(indices.shape, dtype=float)
    for bit in range(num_qubits):
        parity += (indices >> bit) & 1
    return np.mod(parity, 2)


class VariationalQuantumClassifier:
    """Binary VQC with a scikit-learn style ``fit`` / ``predict`` surface.

    Args:
        depth: number of ``RealAmplitudes`` repetitions (the trainable layers).
        circuit_config: feature-map and ansatz structure.
        training_config: optimiser settings.
        noise_config: when enabled, training and inference run on Aer with a
            depolarising noise model instead of the exact statevector backend.
        seed: seeds the weight initialisation and the simulator.
        callback: optional ``(iteration, theta, loss)`` hook, used by the CLI
            for progress logging.
    """

    def __init__(
        self,
        depth: int = 2,
        circuit_config: CircuitConfig | None = None,
        training_config: TrainingConfig | None = None,
        noise_config: NoiseConfig | None = None,
        seed: int = 42,
        callback=None,
    ) -> None:
        self.depth = int(depth)
        self.circuit_config = circuit_config or CircuitConfig()
        self.training_config = training_config or TrainingConfig()
        self.noise_config = noise_config
        self.seed = int(seed)
        self.callback = callback

        self.bundle: CircuitBundle | None = None
        self.backend: CircuitBackend | None = None
        self.weights: np.ndarray | None = None
        self.initial_weights: np.ndarray | None = None
        self.history = TrainingHistory()
        #: Name of the backend the weights were actually trained on. Differs
        #: from ``backend.name`` for models produced by :meth:`transfer_to`.
        self.trained_on: str | None = None
        self._parity: np.ndarray | None = None

    # ------------------------------------------------------------- setup --
    def _initialise(self, n_features: int) -> None:
        self.bundle = build_circuit(n_features, self.circuit_config, depth=self.depth)
        self.backend = make_backend(self.bundle, self.noise_config, seed=self.seed)
        self._parity = parity_readout(self.bundle.num_qubits)

        rng = np.random.default_rng(self.seed)
        # A small symmetric window around zero: large random angles start the
        # optimiser deep in a barren plateau where COBYLA makes no progress.
        self.initial_weights = rng.uniform(-0.1, 0.1, size=self.bundle.num_weights)
        self.weights = self.initial_weights.copy()

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None and self.backend is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("classifier is not fitted; call fit() first")

    # ----------------------------------------------------------- forward --
    def _probability_of_class_one(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        assert self.backend is not None and self._parity is not None
        distribution = self.backend.probabilities(x, theta)
        return distribution @ self._parity

    def _loss(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(self._probability_of_class_one(x, theta), _EPS, 1 - _EPS)
        cross_entropy = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        penalty = self.training_config.l2_regularization * float(np.dot(theta, theta))
        return float(cross_entropy + penalty)

    # -------------------------------------------------------------- fit ---
    def fit(self, x: np.ndarray, y: np.ndarray) -> "VariationalQuantumClassifier":
        """Train the circuit weights on ``(x, y)`` with labels in ``{0, 1}``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")
        if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
            raise ValueError("labels must be binary 0/1")

        self._initialise(x.shape[1])
        assert self.backend is not None and self.initial_weights is not None
        self.backend.reset_counters()
        self.history = TrainingHistory()

        start = time.perf_counter()

        def objective(theta: np.ndarray) -> float:
            value = self._loss(theta, x, y)
            self.history.loss.append(value)
            self.history.elapsed.append(time.perf_counter() - start)
            self.history.n_iterations += 1
            if self.callback is not None:
                self.callback(self.history.n_iterations, theta, value)
            if self.history.n_iterations % 25 == 0:
                LOGGER.debug(
                    "  iteration %3d | loss %.5f", self.history.n_iterations, value
                )
            return value

        result = minimize(
            objective,
            self.initial_weights,
            method=self.training_config.optimizer,
            tol=self.training_config.tol,
            options={
                "maxiter": self.training_config.maxiter,
                "rhobeg": self.training_config.initial_step,
                "disp": False,
            },
        )

        self.weights = np.asarray(result.x, dtype=float)
        self.history.training_time = time.perf_counter() - start
        self.history.converged = bool(result.success)
        self.history.message = str(result.message)
        self.history.final_loss = float(result.fun)
        self.history.circuit_executions = int(self.backend.circuit_executions)

        LOGGER.info(
            "VQC(depth=%d, backend=%s) trained in %.2fs over %d iterations, final loss %.4f",
            self.depth,
            self.backend.name,
            self.history.training_time,
            self.history.n_iterations,
            self.history.final_loss,
        )
        return self

    # -------------------------------------------------------- inference ---
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """``(n_samples, 2)`` array of class probabilities."""
        self._check_fitted()
        assert self.weights is not None
        p1 = self._probability_of_class_one(np.atleast_2d(np.asarray(x, float)), self.weights)
        return np.column_stack([1.0 - p1, p1])

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Signed margin ``P(y=1) - 0.5``, handy for ROC curves."""
        return self.predict_proba(x)[:, 1] - 0.5

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Hard 0/1 predictions."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Plain accuracy, matching the scikit-learn estimator contract."""
        return float(np.mean(self.predict(x) == np.asarray(y).ravel()))

    def transfer_to(
        self,
        noise_config: NoiseConfig | None,
        seed: int | None = None,
    ) -> "VariationalQuantumClassifier":
        """Copy of this fitted model that runs inference on a different backend.

        This is the "train in simulation, deploy on hardware" scenario: the
        weights are carried over untouched and only the evaluation backend
        changes, so the accuracy difference isolates the effect of noise on
        inference from the effect of noise on training.
        """
        self._check_fitted()
        assert self.bundle is not None and self.weights is not None

        clone = VariationalQuantumClassifier(
            depth=self.depth,
            circuit_config=self.circuit_config,
            training_config=self.training_config,
            noise_config=noise_config,
            seed=self.seed if seed is None else seed,
        )
        clone.bundle = self.bundle
        clone._parity = self._parity
        clone.initial_weights = self.initial_weights
        clone.weights = self.weights.copy()
        # The weights were paid for on the original backend; keep its timings
        # so reports do not credit this clone with free training.
        clone.history = self.history
        clone.backend = make_backend(self.bundle, noise_config, seed=clone.seed)
        clone.trained_on = self.backend.name if self.backend else "unknown"
        return clone

    # ----------------------------------------------------------- report ---
    def describe(self) -> dict[str, object]:
        """Structural and backend metadata for the results file."""
        self._check_fitted()
        assert self.bundle is not None and self.backend is not None
        info: dict[str, object] = {
            "depth": self.depth,
            "optimizer": self.training_config.optimizer,
            "trained_on": self.trained_on or self.backend.name,
            "retrained": self.trained_on is None,
        }
        info.update(self.bundle.describe())
        info.update(self.backend.describe())
        return info
