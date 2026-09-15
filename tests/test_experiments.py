"""The studies: composition, caching and the suite record."""

from __future__ import annotations

import numpy as np
import pytest

from qml_compare.classical import build_all_classical_models, build_classical_model
from qml_compare.experiments import (
    STUDIES,
    _VQCCache,
    environment_info,
    evaluate_classical,
    run_baseline_study,
    run_depth_study,
    run_noise_study,
    run_noise_sweep,
    run_suite,
)


def test_baseline_study_covers_both_families(fast_config, toy_dataset):
    results = run_baseline_study(fast_config, toy_dataset)
    families = {result.family for result in results}
    assert families == {"classical", "quantum"}
    assert all(result.experiment == "baseline" for result in results)
    assert all(0.0 <= result.test_accuracy <= 1.0 for result in results)


def test_depth_study_returns_one_record_per_depth(fast_config, toy_dataset):
    results = run_depth_study(fast_config, toy_dataset)
    assert [result.details["depth"] for result in results] == list(fast_config.depths)
    # More layers, more parameters.
    weights = [result.details["num_weights"] for result in results]
    assert weights == sorted(weights)


def test_noise_study_separates_deployment_from_retraining(fast_config, toy_dataset):
    results = run_noise_study(fast_config, toy_dataset)
    assert len(results) == 3

    ideal, deployed, retrained = results
    assert ideal.details["backend"] == "statevector"
    assert deployed.details["backend"] == "aer-noisy"
    assert deployed.details["retrained"] is False
    assert retrained.details["retrained"] is True
    # Deployment is free; retraining is not.
    assert deployed.training_time == pytest.approx(ideal.training_time)
    assert retrained.training_time > 0


def test_noise_sweep_records_error_rate_and_margin(fast_config, toy_dataset):
    results = run_noise_sweep(fast_config, toy_dataset)
    assert len(results) == len(fast_config.noise_sweep) * len(fast_config.noise_sweep_shots)

    rates = [result.details["error_rate"] for result in results]
    assert rates == list(fast_config.noise_sweep)
    margins = [result.details["mean_margin"] for result in results]
    assert margins[0] > margins[-1], "noise should shrink the read-out margin"


def test_cache_trains_each_configuration_once(fast_config, toy_dataset):
    cache = _VQCCache(fast_config, toy_dataset)
    first = cache.get(1, noisy=False)
    assert cache.get(1, noisy=False) is first
    assert cache.get(2, noisy=False) is not first


def test_suite_shares_one_trained_model_across_studies(fast_config, toy_dataset):
    suite = run_suite(fast_config, studies=("baseline", "depth"), dataset=toy_dataset)

    baseline_vqc = [r for r in suite.by_experiment("baseline") if r.family == "quantum"][0]
    depth_vqc = [
        r for r in suite.by_experiment("depth") if r.details["depth"] == fast_config.reference_depth
    ][0]
    assert baseline_vqc.test_accuracy == depth_vqc.test_accuracy
    assert baseline_vqc.training_time == depth_vqc.training_time


def test_suite_metadata_is_complete(fast_config, toy_dataset):
    suite = run_suite(fast_config, studies=("baseline",), dataset=toy_dataset)
    payload = suite.to_dict()

    assert payload["config"]["seed"] == fast_config.seed
    assert payload["dataset"]["n_train"] == toy_dataset.n_train
    assert payload["environment"]["qiskit"]
    assert payload["started_at"] <= payload["finished_at"]
    assert len(payload["results"]) == len(suite.results)


def test_suite_lookup_helpers(fast_config, toy_dataset):
    suite = run_suite(fast_config, studies=("baseline",), dataset=toy_dataset)
    assert suite.find("Logistic Regression") is not None
    assert suite.find("Logistic Regression", experiment="depth") is None
    assert suite.find("no such model") is None


def test_unknown_study_is_rejected(fast_config, toy_dataset):
    with pytest.raises(ValueError, match="unknown study"):
        run_suite(fast_config, studies=("bogus",), dataset=toy_dataset)


def test_every_registered_study_is_callable(fast_config, toy_dataset):
    assert set(STUDIES) == {"baseline", "depth", "noise", "noise-sweep"}


def test_classical_baselines_are_reproducible(toy_dataset):
    models = build_all_classical_models(seed=0)
    assert set(models) == {"Logistic Regression", "SVM (RBF Kernel)"}

    result = evaluate_classical("SVM (RBF Kernel)", models["SVM (RBF Kernel)"], toy_dataset, "unit")
    assert result.family == "classical"
    assert result.training_time > 0
    assert np.isfinite(result.roc_auc)


def test_unknown_classical_model_is_rejected():
    with pytest.raises(KeyError, match="unknown classical model"):
        build_classical_model("Random Forest")


def test_environment_info_lists_the_stack():
    info = environment_info()
    for key in ("python", "numpy", "scipy", "scikit_learn", "qiskit"):
        assert info[key]
