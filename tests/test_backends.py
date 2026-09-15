"""Backend behaviour: normalisation, caching, agreement and noise."""

from __future__ import annotations

import numpy as np
import pytest

from qml_compare.backends import AerBackend, StatevectorBackend, build_noise_model, make_backend
from qml_compare.circuits import build_circuit
from qml_compare.config import CircuitConfig, NoiseConfig


@pytest.fixture(scope="module")
def bundle():
    return build_circuit(3, CircuitConfig(feature_map_reps=1), depth=1)


@pytest.fixture(scope="module")
def samples():
    generator = np.random.default_rng(4)
    return generator.uniform(-1, 1, size=(6, 3))


@pytest.fixture(scope="module")
def weights(bundle):
    return np.random.default_rng(5).uniform(-np.pi, np.pi, size=bundle.num_weights)


def test_probabilities_are_normalised(bundle, samples, weights):
    probabilities = StatevectorBackend(bundle).probabilities(samples, weights)
    assert probabilities.shape == (6, 2**bundle.num_qubits)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert probabilities.min() >= 0.0


def test_statevector_and_aer_agree_without_noise(bundle, samples, weights):
    exact = StatevectorBackend(bundle).probabilities(samples, weights)
    aer = AerBackend(bundle, NoiseConfig(enabled=False, shots=0), seed=1)
    np.testing.assert_allclose(aer.probabilities(samples, weights), exact, atol=1e-8)


def test_shot_sampling_converges_to_the_exact_distribution(bundle, samples, weights):
    exact = StatevectorBackend(bundle).probabilities(samples, weights)
    sampled = AerBackend(bundle, NoiseConfig(enabled=False, shots=20000), seed=1).probabilities(
        samples, weights
    )
    # 20k shots: the standard error per outcome is well under 0.005.
    assert np.abs(sampled - exact).max() < 0.03


def test_encoded_states_are_cached(bundle, samples, weights):
    backend = StatevectorBackend(bundle)
    first = backend.encoded_states(samples)
    assert backend.encoded_states(samples) is first, "same batch should hit the cache"

    backend.clear_cache()
    assert backend.encoded_states(samples) is not first


def test_cache_is_keyed_on_the_data_not_the_object(bundle, samples, weights):
    backend = StatevectorBackend(bundle)
    baseline = backend.probabilities(samples, weights)
    shifted = backend.probabilities(samples + 0.1, weights)
    assert not np.allclose(baseline, shifted)


def test_execution_counter_tracks_circuits(bundle, samples, weights):
    backend = StatevectorBackend(bundle)
    backend.probabilities(samples, weights)
    backend.probabilities(samples, weights)
    assert backend.circuit_executions == 2 * len(samples)

    backend.reset_counters()
    assert backend.circuit_executions == 0


def test_noise_moves_the_distribution_towards_uniform(bundle, samples, weights):
    exact = StatevectorBackend(bundle).probabilities(samples, weights)
    noisy = AerBackend(
        bundle, NoiseConfig(enabled=True, one_qubit_error=0.1, two_qubit_error=0.1, shots=0), seed=1
    ).probabilities(samples, weights)

    uniform = 1.0 / exact.shape[1]
    assert np.abs(noisy - uniform).mean() < np.abs(exact - uniform).mean()
    np.testing.assert_allclose(noisy.sum(axis=1), 1.0, atol=1e-8)


def test_make_backend_picks_the_cheap_path_when_noiseless(bundle):
    assert isinstance(make_backend(bundle, None), StatevectorBackend)
    assert isinstance(make_backend(bundle, NoiseConfig(enabled=False)), StatevectorBackend)
    assert isinstance(make_backend(bundle, NoiseConfig(enabled=True)), AerBackend)


def test_noise_model_covers_one_and_two_qubit_gates():
    model = build_noise_model(NoiseConfig(enabled=True, one_qubit_error=0.01, two_qubit_error=0.02))
    assert "cx" in model.noise_instructions
    assert any(gate in model.noise_instructions for gate in ("u", "rz", "sx"))


def test_mismatched_input_sizes_are_rejected(bundle, weights):
    backend = StatevectorBackend(bundle)
    with pytest.raises(ValueError, match="features"):
        backend.probabilities(np.zeros((2, 5)), weights)
    with pytest.raises(ValueError, match="weights"):
        backend.probabilities(np.zeros((2, 3)), np.zeros(99))


def test_describe_reports_the_backend_identity(bundle):
    noisy = AerBackend(bundle, NoiseConfig(enabled=True, shots=0), seed=1).describe()
    assert noisy["backend"] == "aer-noisy"
    assert noisy["method"] == "density_matrix"
    assert noisy["shots"] == "exact"
