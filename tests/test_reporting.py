"""Report rendering and the artefacts written to disk."""

from __future__ import annotations

import json

import pytest

from qml_compare.experiments import run_suite
from qml_compare.reporting import (
    baseline_table,
    depth_table,
    headline_findings,
    noise_sweep_table,
    noise_table,
    render_report,
    results_dataframe,
    save_suite,
    summarise_to_console,
)


@pytest.fixture(scope="module")
def suite(toy_dataset):
    from qml_compare.config import CircuitConfig, ExperimentConfig, NoiseConfig, TrainingConfig

    config = ExperimentConfig(
        seed=3,
        depths=(1, 2),
        reference_depth=1,
        noise_sweep=(0.0, 0.05),
        noise_sweep_shots=(0,),
        circuit=CircuitConfig(feature_map_reps=1, depth=1),
        training=TrainingConfig(maxiter=10),
        noise=NoiseConfig(one_qubit_error=0.05, two_qubit_error=0.05, shots=0),
    )
    return run_suite(config, dataset=toy_dataset)


def test_dataframe_has_one_row_per_result(suite):
    frame = results_dataframe(suite)
    assert len(frame) == len(suite.results)
    assert {"experiment", "model", "test_accuracy", "training_time_s"} <= set(frame.columns)


def test_tables_render_for_every_study(suite):
    for renderer in (baseline_table, depth_table, noise_table, noise_sweep_table):
        table = renderer(suite)
        assert table.startswith("|")
        assert "| --- |" in table


def test_tables_are_empty_when_a_study_did_not_run(toy_dataset, fast_config):
    partial = run_suite(fast_config, studies=("baseline",), dataset=toy_dataset)
    assert depth_table(partial) == ""
    assert noise_table(partial) == ""


def test_headline_findings_are_sentences(suite):
    findings = headline_findings(suite)
    assert len(findings) >= 3
    assert all(text.endswith(".") for text in findings)


def test_report_names_the_authors_and_the_studies(suite):
    report = render_report(suite)
    assert "Anurag Jha and Sandesh Dhital" in report
    for heading in ("## 1. Classical baselines", "## 2. Circuit depth", "## 3. Hardware noise"):
        assert heading in report
    assert suite.environment["qiskit"] in report


def test_save_suite_writes_all_three_artefacts(suite, tmp_path):
    paths = save_suite(suite, tmp_path)
    assert set(paths) == {"json", "csv", "report"}
    assert all(path.exists() for path in paths.values())

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(payload["results"]) == len(suite.results)
    assert payload["results"][0]["model"]

    csv_lines = paths["csv"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == len(suite.results) + 1  # header


def test_console_summary_is_plain_text(suite):
    text = summarise_to_console(suite)
    assert "model" in text
    assert "\n" in text


def test_suite_round_trips_through_json(suite, tmp_path):
    """A written results.json must rebuild into an equivalent suite."""
    from qml_compare.experiments import ExperimentSuite

    path = save_suite(suite, tmp_path)["json"]
    restored = ExperimentSuite.from_json(path)

    assert len(restored.results) == len(suite.results)
    assert [r.model for r in restored.results] == [r.model for r in suite.results]
    assert restored.dataset == suite.dataset
    assert render_report(restored) == render_report(suite)
