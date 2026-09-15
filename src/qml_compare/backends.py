"""Circuit evaluation backends.

Both backends answer the same question -- given a batch of samples ``X`` and
one weight vector ``theta``, what is the measurement distribution over the
computational basis? -- but they trade accuracy for realism in opposite ways.

``StatevectorBackend``
    Exact, noiseless, and fast. Because the feature map does not depend on the
    weights, the encoded states ``|psi(x)>`` are computed once and cached. Each
    optimiser iteration then only builds the ``2**n x 2**n`` ansatz unitary and
    applies it to the whole cached batch with a single matrix product. This is
    what makes a 200-iteration COBYLA run finish in seconds instead of minutes.

``AerBackend``
    Density-matrix simulation through Aer with a depolarising noise model,
    read out either exactly or by sampling shots. The caching trick does not
    apply (a noisy channel is not a unitary), so the whole batch is submitted
    as one parameter-bound job per iteration, which keeps the Python-side
    overhead to a single call.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector

from qml_compare.circuits import CircuitBundle, measured_circuit
from qml_compare.config import NoiseConfig
from qml_compare.logging_utils import get_logger

LOGGER = get_logger(__name__)


class CircuitBackend(ABC):
    """Common interface used by the classifier to evaluate a circuit."""

    name: str = "abstract"

    def __init__(self, bundle: CircuitBundle) -> None:
        self.bundle = bundle
        self.num_qubits = bundle.num_qubits
        self.dim = 2**bundle.num_qubits
        self.circuit_executions = 0

    @abstractmethod
    def probabilities(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return an ``(n_samples, 2**n_qubits)`` array of basis-state probabilities."""

    def reset_counters(self) -> None:
        """Zero the executed-circuit counter (reported per experiment)."""
        self.circuit_executions = 0

    def describe(self) -> dict[str, Any]:
        """Backend metadata stored with every result record."""
        return {"backend": self.name, "num_qubits": self.num_qubits}

    def _validate(self, x: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coerce inputs to 2-D float arrays and check them against the circuit."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        theta = np.asarray(theta, dtype=float).ravel()
        if x.shape[1] != len(self.bundle.input_parameters):
            raise ValueError(
                f"expected {len(self.bundle.input_parameters)} features, got {x.shape[1]}"
            )
        if theta.size != self.bundle.num_weights:
            raise ValueError(f"expected {self.bundle.num_weights} weights, got {theta.size}")
        return x, theta


class StatevectorBackend(CircuitBackend):
    """Exact simulation with cached feature-map states."""

    name = "statevector"

    def __init__(self, bundle: CircuitBundle) -> None:
        super().__init__(bundle)
        self._cache: dict[bytes, np.ndarray] = {}

    def encoded_states(self, x: np.ndarray) -> np.ndarray:
        """``(n_samples, 2**n)`` matrix of encoded states, memoised per batch.

        The cache key is the raw bytes of the batch, so the training set is
        encoded exactly once no matter how many optimiser iterations run.
        """
        key = x.tobytes() + repr(x.shape).encode()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        params = self.bundle.input_parameters
        states = np.empty((x.shape[0], self.dim), dtype=complex)
        for row, sample in enumerate(x):
            bound = self.bundle.feature_map.assign_parameters(dict(zip(params, sample)))
            states[row] = Statevector.from_instruction(bound).data
        self._cache[key] = states
        LOGGER.debug("Cached %d encoded states", x.shape[0])
        return states

    def ansatz_unitary(self, theta: np.ndarray) -> np.ndarray:
        """Dense unitary of the ansatz for one weight vector."""
        bound = self.bundle.ansatz.assign_parameters(
            dict(zip(self.bundle.weight_parameters, theta))
        )
        return np.asarray(Operator(bound).data)

    def probabilities(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x, theta = self._validate(x, theta)
        states = self.encoded_states(x)
        unitary = self.ansatz_unitary(theta)
        # states @ U.T applies U to every row-vector state in one BLAS call.
        evolved = states @ unitary.T
        self.circuit_executions += x.shape[0]
        return np.abs(evolved) ** 2

    def clear_cache(self) -> None:
        """Drop memoised encoded states (used by the tests)."""
        self._cache.clear()


class AerBackend(CircuitBackend):
    """Simulation through ``qiskit_aer``, with optional depolarising noise.

    Two read-out modes, chosen by ``NoiseConfig.shots``:

    ``shots = 0``
        Exact probabilities read off the density matrix. This is the
        infinite-shot limit, so the measured effect is decoherence alone.
    ``shots > 0``
        The density matrix is sampled, adding finite-sampling noise on top of
        the gate noise. Realistic, but roughly four times slower per iteration
        and it makes the loss surface stochastic.

    The density-matrix method matters for speed as well as clarity: with a
    noise model, Aer's default method runs one stochastic trajectory *per
    shot*, while ``density_matrix`` evolves the state once and reads or samples
    the result -- about 8x faster per optimiser iteration here.
    """

    name = "aer"

    def __init__(self, bundle: CircuitBundle, noise: NoiseConfig, seed: int = 42) -> None:
        super().__init__(bundle)
        from qiskit_aer import AerSimulator

        self.noise_config = noise
        self.shots = max(0, int(noise.shots))
        self.seed = int(seed)
        self.noise_model = build_noise_model(noise) if noise.enabled else None
        self.simulator = AerSimulator(
            method="density_matrix" if noise.enabled else "statevector",
            noise_model=self.noise_model,
            seed_simulator=self.seed,
            # One batched job holds hundreds of circuits; spreading them over
            # cores is where most of the wall-clock saving comes from.
            max_parallel_experiments=os.cpu_count() or 1,
        )

        if self.shots:
            circuit = measured_circuit(bundle)
        else:
            circuit = bundle.circuit.copy()
            circuit.save_probabilities()
        # Transpiling once keeps the per-iteration cost to binding parameters.
        self._transpiled = transpile(
            circuit,
            self.simulator,
            optimization_level=1,
            seed_transpiler=self.seed,
        )
        self._order = list(self._transpiled.parameters)
        self.name = "aer-noisy" if noise.enabled else "aer-ideal"

    def _parameter_binds(self, x: np.ndarray, theta: np.ndarray) -> dict[Any, list[float]]:
        """One value list per circuit parameter, covering the whole batch."""
        n_samples = x.shape[0]
        weights = dict(zip(self.bundle.weight_parameters, theta))
        binds: dict[Any, list[float]] = {}
        for parameter in self._order:
            if parameter in weights:
                binds[parameter] = [float(weights[parameter])] * n_samples
            else:
                column = self.bundle.input_parameters.index(parameter)
                binds[parameter] = [float(value) for value in x[:, column]]
        return binds

    def probabilities(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x, theta = self._validate(x, theta)
        n_samples = x.shape[0]

        result = self.simulator.run(
            self._transpiled,
            parameter_binds=[self._parameter_binds(x, theta)],
            shots=self.shots or 1,
        ).result()
        self.circuit_executions += n_samples

        if self.shots:
            counts_list = result.get_counts()
            if isinstance(counts_list, dict):
                counts_list = [counts_list]
            return _counts_to_probabilities(counts_list, self.dim, self.shots)

        return np.asarray(
            [
                np.asarray(result.data(index)["probabilities"], dtype=float)
                for index in range(n_samples)
            ]
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "shots": self.shots or "exact",
                "method": self.simulator.options.method,
                "noise_enabled": bool(self.noise_config.enabled),
                "one_qubit_error": self.noise_config.one_qubit_error,
                "two_qubit_error": self.noise_config.two_qubit_error,
                "readout_error": self.noise_config.readout_error,
                "transpiled_depth": int(self._transpiled.depth()),
            }
        )
        return info


def build_noise_model(noise: NoiseConfig):
    """Basic depolarising noise model standing in for NISQ hardware.

    A depolarising channel is attached to every one- and two-qubit gate in the
    transpiled basis, plus optional symmetric readout error. It is a
    deliberately crude model: the noise study is about the order of magnitude
    of a VQC's sensitivity, not about matching one specific device.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    model = NoiseModel()
    one_qubit_basis = ["u1", "u2", "u3", "u", "rz", "sx", "x", "ry", "h", "p"]
    two_qubit_basis = ["cx", "cz", "ecr"]

    if noise.one_qubit_error > 0:
        model.add_all_qubit_quantum_error(
            depolarizing_error(noise.one_qubit_error, 1), one_qubit_basis
        )
    if noise.two_qubit_error > 0:
        model.add_all_qubit_quantum_error(
            depolarizing_error(noise.two_qubit_error, 2), two_qubit_basis
        )
    if noise.readout_error > 0:
        p = float(noise.readout_error)
        model.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return model


def _counts_to_probabilities(counts_list, dim: int, shots: int) -> np.ndarray:
    """Convert Aer bitstring counts into dense probability rows."""
    probabilities = np.zeros((len(counts_list), dim), dtype=float)
    for row, counts in enumerate(counts_list):
        total = sum(counts.values()) or shots
        for bitstring, count in counts.items():
            index = int(bitstring.replace(" ", ""), 2)
            probabilities[row, index] = count / total
    return probabilities


def make_backend(
    bundle: CircuitBundle,
    noise: NoiseConfig | None = None,
    seed: int = 42,
) -> CircuitBackend:
    """Pick the cheapest backend that can honour the requested noise settings.

    Noiseless runs take the exact statevector path; anything with noise enabled
    needs Aer.
    """
    if noise is not None and noise.enabled:
        return AerBackend(bundle, noise, seed=seed)
    return StatevectorBackend(bundle)
