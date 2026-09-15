"""Figures for the results directory.

Matplotlib only, ``Agg`` backend, no seaborn -- the figures have to render on a
headless machine and in CI. Every function takes an already-computed suite and
returns the path it wrote, so plotting can be re-run from ``results.json``
without touching a simulator.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
import numpy as np  # noqa: E402

from qml_compare.data import Dataset  # noqa: E402
from qml_compare.experiments import ExperimentSuite  # noqa: E402
from qml_compare.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

CLASSICAL_COLOR = "#4C6EF5"
QUANTUM_COLOR = "#B197FC"
ACCENT_COLOR = "#F76707"
GRID_KWARGS = {"alpha": 0.25, "linewidth": 0.7}


def _finish(fig, path: Path) -> Path:
    """Tighten, save and close a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.debug("Wrote %s", path.name)
    return path


def plot_model_comparison(suite: ExperimentSuite, output_dir: Path) -> Path | None:
    """Accuracy bars next to training time on a log scale."""
    rows = suite.by_experiment("baseline")
    if not rows:
        return None

    names = [r.model for r in rows]
    accuracies = [r.test_accuracy for r in rows]
    times = [r.training_time for r in rows]
    colors = [CLASSICAL_COLOR if r.family == "classical" else QUANTUM_COLOR for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars = left.bar(names, accuracies, color=colors, edgecolor="white")
    left.set_ylim(0, 1.05)
    left.set_ylabel("Test accuracy")
    left.set_title("Accuracy on held-out data")
    left.grid(axis="y", **GRID_KWARGS)
    left.set_axisbelow(True)
    for bar, value in zip(bars, accuracies):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            fontsize=9,
        )

    bars = right.bar(names, times, color=colors, edgecolor="white")
    right.set_yscale("log")
    right.set_ylabel("Training time (s, log scale)")
    right.set_title("Cost of getting there")
    right.grid(axis="y", **GRID_KWARGS)
    right.set_axisbelow(True)
    for bar, value in zip(bars, times):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:.2f}s",
            ha="center",
            fontsize=9,
        )

    for axis in (left, right):
        axis.tick_params(axis="x", rotation=12)

    fig.suptitle("Classical baselines vs. Variational Quantum Classifier", fontsize=13)
    return _finish(fig, output_dir / "model_comparison.png")


def plot_depth_tradeoff(suite: ExperimentSuite, output_dir: Path) -> Path | None:
    """Accuracy and training time against ansatz depth on twin axes."""
    rows = suite.by_experiment("depth")
    if not rows:
        return None

    depths = [r.details.get("depth", index + 1) for index, r in enumerate(rows)]
    accuracies = [r.test_accuracy for r in rows]
    times = [r.training_time for r in rows]

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(depths, accuracies, "o-", color=QUANTUM_COLOR, linewidth=2.2, label="Test accuracy")
    axis.set_xlabel("Ansatz depth (RealAmplitudes repetitions)")
    axis.set_ylabel("Test accuracy", color=QUANTUM_COLOR)
    axis.set_xticks(depths)
    axis.grid(**GRID_KWARGS)
    axis.set_axisbelow(True)
    # Headroom so the point labels are not clipped by the axes or the legend.
    span = max(accuracies) - min(accuracies) or 0.01
    axis.set_ylim(min(accuracies) - 0.25 * span, max(accuracies) + 0.35 * span)
    for depth, accuracy in zip(depths, accuracies):
        axis.annotate(
            f"{accuracy:.1%}",
            (depth, accuracy),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )

    twin = axis.twinx()
    twin.plot(depths, times, "s--", color=ACCENT_COLOR, linewidth=1.8, label="Training time")
    twin.set_ylabel("Training time (s)", color=ACCENT_COLOR)

    handles = axis.get_lines() + twin.get_lines()
    # Upper left: accuracy and time are both lowest at the shallowest depth,
    # so that corner is reliably empty.
    axis.legend(handles, [line.get_label() for line in handles], loc="upper left", fontsize=9)
    axis.set_title("Deeper circuits cost more than they return")
    return _finish(fig, output_dir / "depth_tradeoff.png")


def plot_noise_impact(suite: ExperimentSuite, output_dir: Path) -> Path | None:
    """Accuracy under the three noise conditions."""
    rows = suite.by_experiment("noise")
    if not rows:
        return None

    labels = [r.model.replace(" (", "\n(") for r in rows]
    accuracies = [r.test_accuracy for r in rows]
    colors = [QUANTUM_COLOR, ACCENT_COLOR, CLASSICAL_COLOR][: len(rows)]

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar(labels, accuracies, color=colors, edgecolor="white", width=0.6)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Where noise actually hurts")
    axis.grid(axis="y", **GRID_KWARGS)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            fontsize=10,
        )
    return _finish(fig, output_dir / "noise_impact.png")


def plot_noise_sweep(suite: ExperimentSuite, output_dir: Path) -> Path | None:
    """Accuracy and read-out margin against the gate error rate."""
    rows = suite.by_experiment("noise_sweep")
    if not rows:
        return None

    by_shots: dict[int, list] = {}
    for result in rows:
        by_shots.setdefault(int(result.details.get("sweep_shots", 0)), []).append(result)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for shots, group in sorted(by_shots.items()):
        rates = [100 * r.details.get("error_rate", 0.0) for r in group]
        label = "exact read-out" if shots == 0 else f"{shots} shots"
        left.plot(
            rates,
            [r.test_accuracy for r in group],
            "o-",
            linewidth=2,
            label=label,
        )
        right.plot(
            rates,
            [r.details.get("mean_margin", float("nan")) for r in group],
            "o-",
            linewidth=2,
            label=label,
        )

    left.set_xlabel("Depolarising error per gate (%)")
    left.set_ylabel("Test accuracy")
    left.set_title("Accuracy degrades slowly...")
    left.grid(**GRID_KWARGS)
    left.legend()

    right.set_xlabel("Depolarising error per gate (%)")
    right.set_ylabel("Mean |P(y=1) - 0.5|")
    right.set_title("...but the read-out margin collapses")
    right.grid(**GRID_KWARGS)
    right.legend()

    fig.suptitle("Noise sensitivity of the depth-2 VQC (ideal-trained weights)", fontsize=13)
    return _finish(fig, output_dir / "noise_sweep.png")


def plot_loss_curves(suite: ExperimentSuite, output_dir: Path) -> Path | None:
    """COBYLA convergence traces for every trained VQC.

    Only genuinely trained models are plotted. A model produced by
    ``transfer_to`` carries the loss trace of the run that trained it, so
    plotting it too would draw the same curve again under a label suggesting
    it came from the noisy backend.
    """
    traces = [
        result
        for result in suite.results
        if result.family == "quantum"
        and result.details.get("retrained", True)
        and result.history
        and result.history.get("loss")
    ]
    if not traces:
        return None

    seen: set[tuple[str, int]] = set()
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for result in traces:
        key = (str(result.details.get("trained_on")), int(result.details.get("depth", 0)))
        if key in seen:
            continue
        seen.add(key)
        loss = result.history["loss"]
        axis.plot(range(1, len(loss) + 1), loss, linewidth=1.6, label=result.model)

    axis.set_xlabel("COBYLA objective evaluation")
    axis.set_ylabel("Binary cross-entropy on the training set")
    axis.set_title("Convergence of the hybrid loop")
    axis.grid(**GRID_KWARGS)
    axis.legend(fontsize=8)
    return _finish(fig, output_dir / "loss_curves.png")


def plot_confusion_matrices(suite: ExperimentSuite, output_dir: Path) -> Path | None:
    """Confusion matrices for the baseline comparison."""
    rows = suite.by_experiment("baseline")
    if not rows:
        return None

    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.6))
    axes = np.atleast_1d(axes)
    for axis, result in zip(axes, rows):
        matrix = np.asarray(result.confusion_matrix)
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{result.model}\n{result.test_accuracy:.1%}", fontsize=10)
        axis.set_xticks([0, 1], ["pred 0", "pred 1"])
        axis.set_yticks([0, 1], ["true 0", "true 1"])
        threshold = matrix.max() / 2 if matrix.max() else 0
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(
                column,
                row,
                int(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=12,
            )
    fig.suptitle("Confusion matrices on the held-out split", fontsize=13)
    return _finish(fig, output_dir / "confusion_matrices.png")


def plot_dataset_projection(dataset: Dataset, output_dir: Path) -> Path:
    """First two principal components, coloured by class."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    for axis, (x, y, title) in zip(
        (left, right),
        (
            (dataset.x_train, dataset.y_train, f"Training split (n={dataset.n_train})"),
            (dataset.x_test, dataset.y_test, f"Test split (n={dataset.n_test})"),
        ),
    ):
        for label, color, name in ((0, ACCENT_COLOR, dataset.class_names[0]),
                                   (1, CLASSICAL_COLOR, dataset.class_names[1])):
            mask = y == label
            axis.scatter(x[mask, 0], x[mask, 1], s=18, alpha=0.75, color=color, label=name)
        axis.set_xlabel("PC 1 (scaled)")
        axis.set_ylabel("PC 2 (scaled)")
        axis.set_title(title)
        axis.grid(**GRID_KWARGS)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)

    variance = dataset.total_explained_variance
    fig.suptitle(
        f"Breast Cancer Wisconsin after PCA to {dataset.n_features}D "
        f"({variance:.1%} variance retained)",
        fontsize=13,
    )
    return _finish(fig, output_dir / "dataset_projection.png")


def generate_figures(
    suite: ExperimentSuite,
    output_dir: str | Path,
    dataset: Dataset | None = None,
) -> list[Path]:
    """Write every figure the suite has data for and return their paths."""
    directory = Path(output_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_model_comparison(suite, directory),
        plot_depth_tradeoff(suite, directory),
        plot_noise_impact(suite, directory),
        plot_noise_sweep(suite, directory),
        plot_loss_curves(suite, directory),
        plot_confusion_matrices(suite, directory),
    ]
    if dataset is not None:
        paths.append(plot_dataset_projection(dataset, directory))

    written = [path for path in paths if path is not None]
    LOGGER.info("Wrote %d figures to %s", len(written), directory)
    return written
