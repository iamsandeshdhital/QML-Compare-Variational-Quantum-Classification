"""Command line interface.

    qml-compare run                 # every study, figures and reports
    qml-compare run --study depth   # one study
    qml-compare run --quick         # small optimiser budget, for a smoke test
    qml-compare data                # inspect the preprocessed dataset
    qml-compare circuit --depth 3   # print the circuit and its gate counts
    qml-compare config              # show the resolved configuration
    qml-compare report              # re-render tables and figures from results.json

``python -m qml_compare ...`` is equivalent to the ``qml-compare`` script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qml_compare import __authors__, __version__
from qml_compare.config import DEFAULT_CONFIG_PATH, ExperimentConfig
from qml_compare.logging_utils import configure_logging, get_logger

LOGGER = get_logger(__name__)

STUDY_CHOICES = ("all", "baseline", "depth", "noise", "noise-sweep")
#: A ``--quick`` run trades convergence for turnaround: enough to prove the
#: pipeline works end to end, not enough to quote the numbers.
QUICK_OVERRIDES = {
    "training.maxiter": 40,
    "depths": [1, 2],
    "noise_sweep": [0.0, 0.01, 0.1],
    "noise_sweep_shots": [0],
}


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="qml-compare",
        description=(
            "Variational Quantum Classifier benchmarked against classical baselines. "
            f"By {' and '.join(__authors__)}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"qml-compare {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON configuration file",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the random seed")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run experiments and write results")
    run.add_argument(
        "--study",
        choices=STUDY_CHOICES,
        default="all",
        help="which study to run",
    )
    run.add_argument("--output-dir", type=Path, default=None, help="where to write artefacts")
    run.add_argument("--maxiter", type=int, default=None, help="COBYLA iteration budget")
    run.add_argument("--depth", type=int, default=None, help="override the reference depth")
    run.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="depths for the depth study",
    )
    run.add_argument(
        "--noise-error",
        type=float,
        default=None,
        help="depolarising error per gate used by the noise study",
    )
    run.add_argument(
        "--shots",
        type=int,
        default=None,
        help="shots for noisy simulation; 0 means exact density-matrix read-out",
    )
    run.add_argument("--no-figures", action="store_true", help="skip figure generation")
    run.add_argument("--quick", action="store_true", help="tiny budget, for smoke tests")
    run.set_defaults(handler=_command_run)

    data = subparsers.add_parser("data", help="describe the preprocessed dataset")
    data.add_argument("--components", type=int, default=None, help="PCA components to keep")
    data.set_defaults(handler=_command_data)

    circuit = subparsers.add_parser("circuit", help="print the variational circuit")
    circuit.add_argument("--depth", type=int, default=None, help="ansatz repetitions")
    circuit.add_argument("--decompose", action="store_true", help="show the unrolled circuit")
    circuit.set_defaults(handler=_command_circuit)

    report = subparsers.add_parser(
        "report",
        help="re-render the report and figures from an existing results.json",
    )
    report.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=None,
        help="path to results.json (defaults to <output_dir>/results.json)",
    )
    report.add_argument("--output-dir", type=Path, default=None, help="where to write artefacts")
    report.add_argument("--no-figures", action="store_true", help="tables only")
    report.set_defaults(handler=_command_report)

    config = subparsers.add_parser("config", help="print the resolved configuration")
    config.add_argument("--save", type=Path, default=None, help="write the configuration here")
    config.set_defaults(handler=_command_config)

    return parser


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Resolve the configuration file plus any global overrides."""
    path = Path(args.config)
    config = ExperimentConfig.from_json(path) if path.exists() else ExperimentConfig()
    if args.seed is not None:
        config = config.replace(seed=args.seed)
    return config


# ----------------------------------------------------------------- commands --
def _command_run(args: argparse.Namespace, config: ExperimentConfig) -> int:
    from qml_compare.data import load_dataset
    from qml_compare.experiments import run_suite
    from qml_compare.reporting import save_suite, summarise_to_console

    overrides: dict[str, object] = {}
    if args.quick:
        overrides.update(QUICK_OVERRIDES)
    if args.maxiter is not None:
        overrides["training.maxiter"] = args.maxiter
    if args.depth is not None:
        overrides["reference_depth"] = args.depth
        overrides["circuit.depth"] = args.depth
    if args.depths is not None:
        overrides["depths"] = list(args.depths)
    if args.noise_error is not None:
        overrides["noise.one_qubit_error"] = args.noise_error
        overrides["noise.two_qubit_error"] = args.noise_error
    if args.shots is not None:
        overrides["noise.shots"] = args.shots
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    if overrides:
        config = config.replace(**overrides)

    studies = ("baseline", "depth", "noise", "noise-sweep") if args.study == "all" else (args.study,)
    output_dir = Path(config.output_dir)

    dataset = load_dataset(config.data, seed=config.seed)
    suite = run_suite(config, studies=studies, dataset=dataset)

    paths = save_suite(suite, output_dir)
    config.to_json(output_dir / "config.used.json")

    if not args.no_figures:
        from qml_compare.plotting import generate_figures

        generate_figures(suite, output_dir, dataset=dataset)

    print()
    print(summarise_to_console(suite))
    print()
    print(f"Artefacts written to {output_dir.resolve()}")
    print(f"  report: {paths['report']}")
    return 0


def _command_report(args: argparse.Namespace, config: ExperimentConfig) -> int:
    """Re-render artefacts from a finished run, without re-running anything."""
    from qml_compare.experiments import ExperimentSuite
    from qml_compare.reporting import save_suite, summarise_to_console

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
    results_path = Path(args.results) if args.results else output_dir / "results.json"
    if not results_path.exists():
        LOGGER.error("No results at %s -- run `qml-compare run` first", results_path)
        return 1

    suite = ExperimentSuite.from_json(results_path)
    LOGGER.info("Loaded %d results from %s", len(suite.results), results_path)
    paths = save_suite(suite, output_dir)

    if not args.no_figures:
        from qml_compare.plotting import generate_figures

        # The dataset itself is not stored in results.json, so the projection
        # figure is skipped here; every results-derived figure is redrawn.
        generate_figures(suite, output_dir)

    print()
    print(summarise_to_console(suite))
    print()
    print(f"Re-rendered into {output_dir.resolve()}")
    print(f"  report: {paths['report']}")
    return 0


def _command_data(args: argparse.Namespace, config: ExperimentConfig) -> int:
    from qml_compare.data import load_dataset

    data_config = config.data
    if args.components is not None:
        data_config = type(data_config)(
            n_components=args.components,
            test_size=data_config.test_size,
            feature_range=data_config.feature_range,
            standardize_before_pca=data_config.standardize_before_pca,
        )

    dataset = load_dataset(data_config, seed=config.seed)
    summary = dataset.summary()
    width = max(len(key) for key in summary)
    for key, value in summary.items():
        print(f"{key:<{width}} : {value}")
    return 0


def _command_circuit(args: argparse.Namespace, config: ExperimentConfig) -> int:
    from qml_compare.circuits import build_circuit

    depth = args.depth if args.depth is not None else config.circuit.depth
    bundle = build_circuit(config.data.n_components, config.circuit, depth=depth)

    print(f"Feature map: ZZFeatureMap(reps={config.circuit.feature_map_reps}, "
          f"entanglement={config.circuit.feature_map_entanglement!r}, "
          f"data_map={config.circuit.feature_map_data_map!r})")
    print(f"Ansatz:      RealAmplitudes(reps={depth}, "
          f"entanglement={config.circuit.ansatz_entanglement!r})")
    print()
    print(bundle.circuit.decompose(reps=3) if args.decompose else bundle.circuit)
    print()
    for key, value in bundle.describe().items():
        print(f"{key:<16} : {value}")
    return 0


def _command_config(args: argparse.Namespace, config: ExperimentConfig) -> int:
    import json

    print(json.dumps(config.to_dict(), indent=2))
    if args.save is not None:
        path = config.to_json(args.save)
        print(f"\nWritten to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for both ``qml-compare`` and ``python -m qml_compare``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = load_config(args)
    except (OSError, ValueError) as error:
        parser.error(f"could not load configuration: {error}")
        return 2

    return int(args.handler(args, config))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
