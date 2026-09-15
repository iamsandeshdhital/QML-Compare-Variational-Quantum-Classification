"""The classifier itself: read-out, training, prediction and transfer."""

from __future__ import annotations

import numpy as np
import pytest

from qml_compare.config import CircuitConfig, NoiseConfig, TrainingConfig
from qml_compare.vqc import VariationalQuantumClassifier, parity_readout

FAST_CIRCUIT = CircuitConfig(feature_map_reps=1, depth=1)
FAST_TRAINING = TrainingConfig(maxiter=25)


@pytest.fixture(scope="module")
def trained(toy_dataset):
    classifier = VariationalQuantumClassifier(
        depth=1, circuit_config=FAST_CIRCUIT, training_config=FAST_TRAINING, seed=3
    )
    return classifier.fit(toy_dataset.x_train, toy_dataset.y_train)


def test_parity_readout_selects_odd_bitstrings():
    parity = parity_readout(3)
    assert parity.shape == (8,)
    # 0b000 -> even, 0b001 -> odd, 0b011 -> even, 0b111 -> odd
    assert list(parity) == [0, 1, 1, 0, 1, 0, 0, 1]


def test_parity_of_a_distribution_is_a_probability():
    parity = parity_readout(2)
    distribution = np.array([0.1, 0.2, 0.3, 0.4])
    assert 0.0 <= distribution @ parity <= 1.0


def test_fit_learns_a_separable_problem(trained, toy_dataset):
    assert trained.is_fitted
    assert trained.score(toy_dataset.x_train, toy_dataset.y_train) > 0.8
    assert trained.score(toy_dataset.x_test, toy_dataset.y_test) > 0.7


def test_training_reduces_the_loss(trained):
    losses = trained.history.loss
    assert len(losses) > 1
    assert min(losses) < losses[0]
    assert trained.history.final_loss == pytest.approx(min(losses), rel=1e-6)


def test_history_records_timing_and_work(trained, toy_dataset):
    history = trained.history
    assert history.training_time > 0
    assert history.n_iterations == len(history.loss) == len(history.elapsed)
    assert history.circuit_executions == history.n_iterations * len(toy_dataset.x_train)
    assert history.elapsed == sorted(history.elapsed)


def test_predictions_and_probabilities_are_consistent(trained, toy_dataset):
    proba = trained.predict_proba(toy_dataset.x_test)
    predictions = trained.predict(toy_dataset.x_test)

    assert proba.shape == (len(toy_dataset.x_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_array_equal(predictions, (proba[:, 1] >= 0.5).astype(int))
    np.testing.assert_allclose(
        trained.decision_function(toy_dataset.x_test), proba[:, 1] - 0.5, atol=1e-12
    )


def test_prediction_is_deterministic_on_the_exact_backend(trained, toy_dataset):
    np.testing.assert_array_equal(
        trained.predict(toy_dataset.x_test), trained.predict(toy_dataset.x_test)
    )


def test_same_seed_gives_the_same_model(toy_dataset):
    def build():
        return VariationalQuantumClassifier(
            depth=1, circuit_config=FAST_CIRCUIT, training_config=TrainingConfig(maxiter=10), seed=8
        ).fit(toy_dataset.x_train, toy_dataset.y_train)

    np.testing.assert_allclose(build().weights, build().weights)


def test_weight_count_matches_the_ansatz(trained):
    assert trained.weights.shape == (trained.bundle.num_weights,)
    assert trained.bundle.num_weights == 2 * (1 + 1)  # 2 qubits, depth 1


def test_predicting_before_fitting_raises():
    classifier = VariationalQuantumClassifier(depth=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier.predict(np.zeros((1, 2)))


def test_non_binary_labels_are_rejected(toy_dataset):
    classifier = VariationalQuantumClassifier(depth=1, training_config=TrainingConfig(maxiter=2))
    with pytest.raises(ValueError, match="binary"):
        classifier.fit(toy_dataset.x_train, toy_dataset.y_train + 1)


def test_mismatched_lengths_are_rejected(toy_dataset):
    classifier = VariationalQuantumClassifier(depth=1, training_config=TrainingConfig(maxiter=2))
    with pytest.raises(ValueError, match="rows"):
        classifier.fit(toy_dataset.x_train, toy_dataset.y_train[:-3])


def test_callback_sees_every_iteration(toy_dataset):
    seen: list[int] = []
    classifier = VariationalQuantumClassifier(
        depth=1,
        circuit_config=FAST_CIRCUIT,
        training_config=TrainingConfig(maxiter=6),
        seed=1,
        callback=lambda iteration, theta, loss: seen.append(iteration),
    )
    classifier.fit(toy_dataset.x_train, toy_dataset.y_train)
    assert seen == list(range(1, len(seen) + 1))


def test_transfer_keeps_weights_and_swaps_the_backend(trained, toy_dataset):
    deployed = trained.transfer_to(NoiseConfig(enabled=True, one_qubit_error=0.05, shots=0))

    np.testing.assert_allclose(deployed.weights, trained.weights)
    assert deployed.backend.name == "aer-noisy"
    assert deployed.trained_on == "statevector"
    # Training cost stays attributed to the run that actually paid it.
    assert deployed.history.training_time == trained.history.training_time
    assert deployed.describe()["retrained"] is False


def test_noise_shrinks_the_decision_margin(trained, toy_dataset):
    deployed = trained.transfer_to(NoiseConfig(enabled=True, one_qubit_error=0.1, shots=0))
    clean_margin = np.abs(trained.decision_function(toy_dataset.x_test)).mean()
    noisy_margin = np.abs(deployed.decision_function(toy_dataset.x_test)).mean()
    assert noisy_margin < clean_margin


def test_l2_regularisation_pulls_weights_towards_zero(toy_dataset):
    def train(penalty):
        return VariationalQuantumClassifier(
            depth=1,
            circuit_config=FAST_CIRCUIT,
            training_config=TrainingConfig(maxiter=30, l2_regularization=penalty),
            seed=2,
        ).fit(toy_dataset.x_train, toy_dataset.y_train)

    assert np.linalg.norm(train(1.0).weights) < np.linalg.norm(train(0.0).weights)
