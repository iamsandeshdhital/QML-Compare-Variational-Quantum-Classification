"""Configuration loading, overriding and round-tripping."""

from __future__ import annotations

import json

import pytest

from qml_compare.config import DEFAULT_CONFIG_PATH, ExperimentConfig


def test_defaults_match_the_documented_setup():
    config = ExperimentConfig()
    assert config.data.n_components == 4
    assert config.data.feature_range == (-1.0, 1.0)
    assert config.data.test_size == pytest.approx(0.2)
    assert config.training.optimizer == "COBYLA"
    assert config.depths == (1, 2, 3)


def test_shipped_default_config_is_loadable():
    assert DEFAULT_CONFIG_PATH.exists(), "configs/default.json should ship with the package"
    config = ExperimentConfig.from_json(DEFAULT_CONFIG_PATH)
    assert config.reference_depth in config.depths


def test_json_round_trip(tmp_path):
    config = ExperimentConfig().replace(seed=11, **{"circuit.depth": 3})
    path = config.to_json(tmp_path / "config.json")
    assert ExperimentConfig.from_json(path) == config


def test_replace_reaches_nested_sections():
    config = ExperimentConfig()
    updated = config.replace(**{"noise.one_qubit_error": 0.05, "training.maxiter": 7})

    assert updated.noise.one_qubit_error == pytest.approx(0.05)
    assert updated.training.maxiter == 7
    # The original is frozen and must be untouched.
    assert config.noise.one_qubit_error == pytest.approx(0.01)


def test_partial_dict_keeps_remaining_defaults():
    config = ExperimentConfig.from_dict({"seed": 5, "training": {"maxiter": 3}})
    assert config.seed == 5
    assert config.training.maxiter == 3
    assert config.training.optimizer == "COBYLA"


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        ExperimentConfig.from_dict({"training": {"learning_rate": 0.1}})

    with pytest.raises(ValueError, match="unknown key"):
        ExperimentConfig.from_dict({"epochs": 10})


def test_to_dict_is_json_serialisable():
    payload = json.dumps(ExperimentConfig().to_dict())
    assert "output_dir" in json.loads(payload)
