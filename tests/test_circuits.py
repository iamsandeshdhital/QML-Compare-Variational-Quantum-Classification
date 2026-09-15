"""Circuit construction: structure, parameter ordering and the data map."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qml_compare.circuits import (
    build_ansatz,
    build_circuit,
    build_feature_map,
    measured_circuit,
    ordered_parameters,
    product_data_map,
)
from qml_compare.config import CircuitConfig


def test_one_qubit_per_feature():
    for n_features in (2, 3, 4, 6):
        bundle = build_circuit(n_features, CircuitConfig())
        assert bundle.num_qubits == n_features
        assert len(bundle.input_parameters) == n_features


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5])
def test_weight_count_follows_the_real_amplitudes_formula(depth):
    bundle = build_circuit(4, CircuitConfig(), depth=depth)
    assert bundle.num_weights == 4 * (depth + 1)


def test_deeper_ansatz_means_more_entanglers():
    shallow = build_circuit(4, CircuitConfig(), depth=1).describe()
    deep = build_circuit(4, CircuitConfig(), depth=3).describe()
    assert deep["two_qubit_gates"] > shallow["two_qubit_gates"]
    assert deep["circuit_depth"] > shallow["circuit_depth"]


def test_parameters_are_ordered_by_index_not_by_name():
    # 12 features means x[10] and x[11] exist, which sort before x[2] by name.
    feature_map = build_feature_map(12, CircuitConfig())
    names = [parameter.name for parameter in ordered_parameters(feature_map)]
    assert names == [f"x[{index}]" for index in range(12)]


def test_composed_circuit_holds_both_blocks():
    bundle = build_circuit(3, CircuitConfig(), depth=2)
    assert set(bundle.circuit.parameters) == set(bundle.input_parameters) | set(
        bundle.weight_parameters
    )


def test_product_data_map_keeps_phases_small():
    assert product_data_map([0.5]) == pytest.approx(0.5)
    assert product_data_map([0.5, -0.4]) == pytest.approx(-0.2)
    # Every pairwise phase stays inside [-1, 1] for scaled inputs, unlike
    # Qiskit's default map which would give (pi - 0.5)(pi + 0.4) ~ 9.3.
    assert abs(product_data_map([1.0, -1.0])) <= 1.0


def test_qiskit_default_data_map_is_still_selectable():
    config = CircuitConfig(feature_map_data_map="qiskit")
    assert build_feature_map(3, config).num_parameters == 3


def test_unknown_data_map_is_rejected():
    with pytest.raises(ValueError, match="unknown feature_map_data_map"):
        build_feature_map(3, CircuitConfig(feature_map_data_map="nonsense"))


def test_encoding_is_injective_enough_to_separate_inputs():
    """Two different samples must not collapse onto the same state."""
    bundle = build_circuit(3, CircuitConfig())
    params = bundle.input_parameters

    def encode(values):
        bound = bundle.feature_map.assign_parameters(dict(zip(params, values)))
        return Statevector.from_instruction(bound).data

    first = encode([0.3, -0.7, 0.1])
    second = encode([-0.2, 0.5, 0.9])
    assert not np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_measured_circuit_adds_measurements_without_mutating_the_bundle():
    bundle = build_circuit(2, CircuitConfig())
    measured = measured_circuit(bundle)
    assert measured.num_clbits == 2
    assert bundle.circuit.num_clbits == 0


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError, match="n_features"):
        build_feature_map(0, CircuitConfig())
    with pytest.raises(ValueError, match="depth"):
        build_ansatz(2, -1, CircuitConfig())
