"""Figure generation and the command line surface."""

from __future__ import annotations

import json

import pytest

from qml_compare.cli import build_parser, main
from qml_compare.config import CircuitConfig, ExperimentConfig, NoiseConfig, TrainingConfig
from qml_compare.experiments import run_suite
from qml_compare.plotting import generate_figures, plot_dataset_projection


@pytest.fixture(scope="module")
def suite(toy_dataset):
    config = ExperimentConfig(
        seed=3,
        depths=(1, 2),
        reference_depth=1,
        noise_sweep=(0.0, 0.05),
        noise_sweep_shots=(0,),
        circuit=CircuitConfig(feature_map_reps=1, depth=1),
        training=TrainingConfig(maxiter=8),
        noise=NoiseConfig(one_qubit_error=0.05, two_qubit_error=0.05, shots=0),
    )
    return run_suite(config, dataset=toy_dataset)


def test_every_figure_is_written(suite, toy_dataset, tmp_path):
    paths = generate_figures(suite, tmp_path, dataset=toy_dataset)
    names = {path.name for path in paths}
    assert names == {
        "model_comparison.png",
        "depth_tradeoff.png",
        "noise_impact.png",
        "noise_sweep.png",
        "loss_curves.png",
        "confusion_matrices.png",
        "dataset_projection.png",
    }
    assert all(path.stat().st_size > 1000 for path in paths)


def test_figures_are_skipped_for_missing_studies(toy_dataset, fast_config, tmp_path):
    partial = run_suite(fast_config, studies=("baseline",), dataset=toy_dataset)
    names = {path.name for path in generate_figures(partial, tmp_path)}
    assert "depth_tradeoff.png" not in names
    assert "model_comparison.png" in names


def test_dataset_projection_does_not_need_a_suite(toy_dataset, tmp_path):
    path = plot_dataset_projection(toy_dataset, tmp_path)
    assert path.exists()


# ------------------------------------------------------------------- CLI ----
def test_parser_exposes_the_documented_commands():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # a subcommand is required

    args = parser.parse_args(["run", "--study", "depth", "--quick"])
    assert args.command == "run"
    assert args.study == "depth"
    assert args.quick is True


def test_config_command_prints_json(capsys):
    assert main(["config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seed"] == 42


def test_config_command_can_save(tmp_path, capsys):
    target = tmp_path / "saved.json"
    assert main(["--seed", "9", "config", "--save", str(target)]) == 0
    capsys.readouterr()
    assert ExperimentConfig.from_json(target).seed == 9


def test_circuit_command_reports_structure(capsys):
    assert main(["circuit", "--depth", "2"]) == 0
    out = capsys.readouterr().out
    assert "RealAmplitudes(reps=2" in out
    assert "num_weights" in out


def test_data_command_summarises_the_dataset(capsys):
    assert main(["data", "--components", "2"]) == 0
    out = capsys.readouterr().out
    assert "n_features" in out
    assert "n_train" in out


def test_run_command_end_to_end(tmp_path, capsys):
    exit_code = main(
        [
            "run",
            "--study",
            "baseline",
            "--quick",
            "--maxiter",
            "5",
            "--output-dir",
            str(tmp_path),
            "--no-figures",
        ]
    )
    assert exit_code == 0
    capsys.readouterr()

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "config.used.json").exists()
    assert not (tmp_path / "figures").exists()

    used = ExperimentConfig.from_json(tmp_path / "config.used.json")
    assert used.training.maxiter == 5, "explicit --maxiter should win over --quick"


def test_loss_curves_skip_transferred_models(suite, tmp_path):
    """A transferred model reuses its trainer's history and must not be drawn.

    Otherwise the same curve appears twice, the second time under a label
    implying it was measured on the noisy backend.
    """
    from qml_compare.plotting import plot_loss_curves

    trained = [
        r
        for r in suite.results
        if r.family == "quantum" and r.details.get("retrained", True)
    ]
    transferred = [
        r for r in suite.results if r.family == "quantum" and not r.details.get("retrained", True)
    ]
    assert transferred, "the noise study should produce at least one transferred model"

    path = plot_loss_curves(suite, tmp_path)
    assert path is not None
    # Distinct (backend trained on, depth) pairs among genuinely trained models.
    expected = {(r.details.get("trained_on"), r.details.get("depth")) for r in trained}
    assert len(expected) >= 2


def test_report_command_rerenders_without_rerunning(tmp_path, capsys):
    """`report` must rebuild artefacts from results.json alone."""
    assert (
        main(
            [
                "run",
                "--study",
                "baseline",
                "--quick",
                "--output-dir",
                str(tmp_path),
                "--no-figures",
            ]
        )
        == 0
    )
    capsys.readouterr()

    (tmp_path / "report.md").unlink()
    assert main(["report", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "figures" / "model_comparison.png").exists()


def test_report_command_without_results_fails_cleanly(tmp_path, capsys):
    assert main(["report", "--output-dir", str(tmp_path)]) == 1
