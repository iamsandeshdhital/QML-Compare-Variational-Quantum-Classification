"""Serialisation of a finished run.

Three artefacts land in the output directory:

``results.json``   the complete suite, including per-iteration loss traces
``results.csv``    one flat row per evaluated model, for spreadsheets
``report.md``      the tables that go into the engineering documentation

Nothing here recomputes anything -- the numbers come straight from the
:class:`~qml_compare.experiments.ExperimentSuite`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from qml_compare.experiments import ExperimentSuite
from qml_compare.logging_utils import get_logger
from qml_compare.metrics import accuracy_drop, relative_change

LOGGER = get_logger(__name__)

AUTHORS = "Anurag Jha and Sandesh Dhital"


def results_dataframe(suite: ExperimentSuite) -> pd.DataFrame:
    """Flat table of every evaluated model."""
    return pd.DataFrame([result.row() for result in suite.results])


def save_suite(suite: ExperimentSuite, output_dir: str | Path) -> dict[str, Path]:
    """Write JSON, CSV and Markdown artefacts; return the paths by kind."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / "results.json"
    json_path.write_text(
        json.dumps(suite.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )

    csv_path = directory / "results.csv"
    results_dataframe(suite).to_csv(csv_path, index=False)

    report_path = directory / "report.md"
    report_path.write_text(render_report(suite), encoding="utf-8")

    LOGGER.info("Wrote %s, %s and %s", json_path.name, csv_path.name, report_path.name)
    return {"json": json_path, "csv": csv_path, "report": report_path}


# ------------------------------------------------------------------- tables --
def baseline_table(suite: ExperimentSuite) -> str:
    """Classical vs. quantum accuracy and training cost."""
    rows = suite.by_experiment("baseline")
    if not rows:
        return ""
    lines = [
        "| Model | Test Accuracy | F1 | ROC-AUC | Training Time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in rows:
        lines.append(
            f"| {result.model} | {result.test_accuracy:.1%} | {result.f1:.3f} "
            f"| {result.roc_auc:.3f} | {result.training_time:.2f} s |"
        )
    return "\n".join(lines)


def depth_table(suite: ExperimentSuite) -> str:
    """Accuracy, weight count and cost per ansatz depth."""
    rows = suite.by_experiment("depth")
    if not rows:
        return ""
    lines = [
        "| VQC Depth | Trainable Weights | Test Accuracy | Training Time | Accuracy Gain |",
        "| --- | --- | --- | --- | --- |",
    ]
    previous: float | None = None
    for result in rows:
        weights = result.details.get("num_weights", "-")
        if previous is None:
            gain = "baseline"
        else:
            gain = f"{accuracy_drop(previous, result.test_accuracy) * -1:+.1f} pp"
        lines.append(
            f"| {result.details.get('depth', '?')} | {weights} | {result.test_accuracy:.1%} "
            f"| {result.training_time:.2f} s | {gain} |"
        )
        previous = result.test_accuracy
    return "\n".join(lines)


def noise_table(suite: ExperimentSuite) -> str:
    """Ideal vs. noisy inference vs. noise-aware training."""
    rows = suite.by_experiment("noise")
    if not rows:
        return ""
    lines = [
        "| Condition | Test Accuracy | F1 | ROC-AUC | Training Time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in rows:
        lines.append(
            f"| {result.model} | {result.test_accuracy:.1%} | {result.f1:.3f} "
            f"| {result.roc_auc:.3f} | {result.training_time:.2f} s |"
        )
    if len(rows) >= 2:
        drop = accuracy_drop(rows[0].test_accuracy, rows[1].test_accuracy)
        lines.append("")
        lines.append(
            "Accuracy lost when ideal-trained weights are deployed on the noisy "
            f"backend: **{drop:.1f} percentage points**."
        )
    return "\n".join(lines)


def noise_sweep_table(suite: ExperimentSuite) -> str:
    """Accuracy and read-out margin as the gate error rate rises."""
    rows = suite.by_experiment("noise_sweep")
    if not rows:
        return ""
    lines = [
        "| Gate Error Rate | Read-out | Test Accuracy | ROC-AUC | Mean Margin from 0.5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in rows:
        shots = result.details.get("sweep_shots", 0)
        readout = "exact" if not shots else f"{shots} shots"
        lines.append(
            f"| {result.details.get('error_rate', 0.0):.1%} | {readout} "
            f"| {result.test_accuracy:.1%} | {result.roc_auc:.3f} "
            f"| {result.details.get('mean_margin', float('nan')):.3f} |"
        )
    return "\n".join(lines)


def headline_findings(suite: ExperimentSuite) -> list[str]:
    """Two or three sentences a reader can quote without reading the tables."""
    findings: list[str] = []

    baseline = suite.by_experiment("baseline")
    quantum = [r for r in baseline if r.family == "quantum"]
    classical = [r for r in baseline if r.family == "classical"]
    if quantum and classical:
        best = max(classical, key=lambda r: r.test_accuracy)
        vqc = quantum[0]
        slowdown = vqc.training_time / best.training_time if best.training_time else float("inf")
        findings.append(
            f"The VQC reaches {vqc.test_accuracy:.1%} test accuracy against "
            f"{best.test_accuracy:.1%} for the best classical baseline ({best.model}), "
            f"while taking {slowdown:.0f}x longer to train."
        )

    depths = suite.by_experiment("depth")
    if len(depths) >= 2:
        first, last = depths[0], depths[-1]
        best_depth = max(depths, key=lambda r: r.test_accuracy)
        findings.append(
            f"Accuracy peaks at depth {best_depth.details.get('depth', '?')} "
            f"({best_depth.test_accuracy:.1%}); going from depth "
            f"{first.details.get('depth', '?')} to {last.details.get('depth', '?')} changes "
            f"accuracy by {accuracy_drop(first.test_accuracy, last.test_accuracy) * -1:+.1f} pp "
            f"for {relative_change(first.training_time, last.training_time):+.0f}% training time."
        )

    noise = suite.by_experiment("noise")
    if len(noise) >= 2:
        findings.append(
            f"Deploying the ideal-trained weights on the noisy backend costs "
            f"{accuracy_drop(noise[0].test_accuracy, noise[1].test_accuracy):.1f} percentage "
            f"points of accuracy"
            + (
                f", and retraining under the same noise recovers "
                f"{accuracy_drop(noise[1].test_accuracy, noise[2].test_accuracy) * -1:+.1f} pp."
                if len(noise) >= 3
                else "."
            )
        )

    sweep = suite.by_experiment("noise_sweep")
    exact = [r for r in sweep if not r.details.get("sweep_shots")]
    if len(exact) >= 2:
        first, last = exact[0], exact[-1]
        findings.append(
            f"Across the sweep, the read-out margin collapses from "
            f"{first.details.get('mean_margin', float('nan')):.3f} to "
            f"{last.details.get('mean_margin', float('nan')):.3f} while accuracy only falls "
            f"{accuracy_drop(first.test_accuracy, last.test_accuracy):.1f} pp -- noise destroys "
            f"the confidence of a VQC well before it destroys its decisions."
        )
    return findings


def render_report(suite: ExperimentSuite) -> str:
    """Full Markdown report for the results directory."""
    dataset = suite.dataset
    environment = suite.environment

    sections: list[str] = [
        "# QML-Compare — Experiment Report",
        "",
        f"*Generated by the QML-Compare experiment harness. Authors: {AUTHORS}.*",
        "",
        f"- Run started: `{suite.started_at}`",
        f"- Run finished: `{suite.finished_at}`",
        f"- Random seed: `{suite.config.get('seed')}`",
        "",
        "## Headline findings",
        "",
    ]
    sections += [f"{index}. {text}" for index, text in enumerate(headline_findings(suite), 1)]
    sections += [
        "",
        "## Dataset",
        "",
        f"- Breast Cancer Wisconsin, reduced to {dataset['n_features']} principal components",
        f"- {dataset['n_train']} training samples / {dataset['n_test']} test samples",
        f"- Variance retained by PCA: {dataset['total_explained_variance']:.1%}",
        f"- Feature range after scaling: "
        f"[{dataset['feature_min']:.2f}, {dataset['feature_max']:.2f}]",
        "",
        "## 1. Classical baselines vs. VQC",
        "",
        baseline_table(suite),
        "",
        "## 2. Circuit depth",
        "",
        depth_table(suite),
        "",
        "## 3. Hardware noise",
        "",
        noise_table(suite),
        "",
        "## 4. Noise sweep",
        "",
        noise_sweep_table(suite),
        "",
        "## Environment",
        "",
        "| Component | Version |",
        "| --- | --- |",
    ]
    sections += [f"| {key} | {value} |" for key, value in environment.items()]
    sections.append("")
    return "\n".join(sections)


def summarise_to_console(suite: ExperimentSuite) -> str:
    """Compact plain-text summary printed at the end of a CLI run."""
    frame = results_dataframe(suite)
    if frame.empty:
        return "No results."
    columns = ["experiment", "model", "test_accuracy", "f1", "training_time_s"]
    return frame[columns].to_string(index=False)


def load_suite_dict(path: str | Path) -> dict[str, Any]:
    """Read a previously written ``results.json``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
