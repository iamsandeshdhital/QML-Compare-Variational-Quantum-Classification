"""Parameterised quantum circuit construction.

The classifier circuit is the standard two-block variational layout:

    |0..0> --[ ZZFeatureMap(x) ]--[ RealAmplitudes(theta) ]-- measure

The feature map is fixed (it only depends on the input sample) and the ansatz
carries the trainable weights. Keeping the two blocks as separate objects is
what lets :mod:`qml_compare.backends` cache the encoded states and re-apply
only the ansatz on every optimiser iteration.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from functools import reduce

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qml_compare.config import CircuitConfig

try:  # Qiskit >= 1.3 ships plain-function builders; the classes are deprecated.
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
except ImportError:  # pragma: no cover - only hit on older Qiskit releases
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

    def zz_feature_map(**kwargs):
        return ZZFeatureMap(**kwargs)

    def real_amplitudes(num_qubits, reps, entanglement, parameter_prefix):
        return RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
        )


@dataclass(frozen=True)
class CircuitBundle:
    """Feature map, ansatz and the composed circuit, with their parameter lists."""

    feature_map: QuantumCircuit
    ansatz: QuantumCircuit
    circuit: QuantumCircuit
    input_parameters: ParameterVector | list
    weight_parameters: ParameterVector | list

    @property
    def num_qubits(self) -> int:
        return int(self.circuit.num_qubits)

    @property
    def num_weights(self) -> int:
        return len(self.weight_parameters)

    def describe(self) -> dict[str, object]:
        """Structural facts recorded alongside every experiment result."""
        decomposed = self.circuit.decompose(reps=3)
        ops = {str(k): int(v) for k, v in decomposed.count_ops().items()}
        return {
            "num_qubits": self.num_qubits,
            "num_weights": self.num_weights,
            "circuit_depth": int(decomposed.depth()),
            "gate_counts": ops,
            "two_qubit_gates": int(sum(v for k, v in ops.items() if k in {"cx", "cz", "ecr"})),
        }


def product_data_map(x):
    """Interaction phase ``prod(x)`` for the ZZ terms.

    Qiskit's default map uses ``prod(pi - x_i)``. With features scaled to
    ``[-1, 1]`` that puts every pairwise phase between roughly 9 and 34
    radians, so the encoded state oscillates wildly over the data range and the
    optimiser cannot find structure -- measured at ~82% test accuracy against
    ~92% for the plain product below. The product keeps the phase inside
    ``[-2, 2]`` radians, where nearby samples stay nearby in Hilbert space.

    Qiskit calls this with symbolic ``ParameterExpression`` objects while it
    builds the circuit, so the arithmetic here has to stay symbolic -- no
    ``float()`` casts.
    """
    return reduce(operator.mul, x)


#: Name -> data map used by :func:`build_feature_map`. ``qiskit`` selects the
#: library default (``prod(pi - x_i)``) and is kept for comparison runs.
DATA_MAPS = {"product": product_data_map, "qiskit": None}


def build_feature_map(n_features: int, config: CircuitConfig) -> QuantumCircuit:
    """Second-order Pauli-Z evolution encoder (``ZZFeatureMap``).

    One qubit per feature. ``reps`` repetitions of Hadamards, single-qubit
    Z rotations and entangling ZZ interactions push the classical vector into a
    correlated region of Hilbert space that a linear model cannot reproduce.
    """
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    try:
        data_map = DATA_MAPS[config.feature_map_data_map]
    except KeyError:
        known = ", ".join(sorted(DATA_MAPS))
        raise ValueError(
            f"unknown feature_map_data_map {config.feature_map_data_map!r}; expected: {known}"
        ) from None

    kwargs = {
        "feature_dimension": n_features,
        "reps": config.feature_map_reps,
        "entanglement": config.feature_map_entanglement,
        "parameter_prefix": "x",
    }
    if data_map is not None:
        kwargs["data_map_func"] = data_map
    return zz_feature_map(**kwargs)


def build_ansatz(n_qubits: int, depth: int, config: CircuitConfig) -> QuantumCircuit:
    """Hardware-efficient ``RealAmplitudes`` ansatz with ``depth`` repetitions.

    Each repetition is a layer of trainable ``Ry`` rotations followed by CNOT
    entanglers, so the weight count is ``n_qubits * (depth + 1)``.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    return real_amplitudes(
        num_qubits=n_qubits,
        reps=depth,
        entanglement=config.ansatz_entanglement,
        parameter_prefix="theta",
    )


def build_circuit(n_features: int, config: CircuitConfig, depth: int | None = None) -> CircuitBundle:
    """Build the full encode-then-train circuit for ``n_features`` inputs."""
    depth = config.depth if depth is None else depth
    feature_map = build_feature_map(n_features, config)
    ansatz = build_ansatz(feature_map.num_qubits, depth, config)

    circuit = QuantumCircuit(feature_map.num_qubits, name=f"vqc_d{depth}")
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    return CircuitBundle(
        feature_map=feature_map,
        ansatz=ansatz,
        circuit=circuit,
        input_parameters=ordered_parameters(feature_map),
        weight_parameters=ordered_parameters(ansatz),
    )


def ordered_parameters(circuit: QuantumCircuit) -> list:
    """Parameters of ``circuit`` in index order rather than alphabetical order.

    ``QuantumCircuit.parameters`` sorts by name, which puts ``x[10]`` before
    ``x[2]``. Feature and weight vectors are positional, so we sort on the
    bracketed index instead and fall back to the name when there is none.
    """

    def sort_key(parameter) -> tuple[str, int, str]:
        match = re.match(r"^(.*)\[(\d+)\]$", parameter.name)
        if match:
            return (match.group(1), int(match.group(2)), "")
        return (parameter.name, -1, parameter.name)

    return sorted(circuit.parameters, key=sort_key)


def measured_circuit(bundle: CircuitBundle) -> QuantumCircuit:
    """Copy of the bundle circuit with a full computational-basis measurement."""
    circuit = bundle.circuit.copy()
    circuit.measure_all()
    return circuit
