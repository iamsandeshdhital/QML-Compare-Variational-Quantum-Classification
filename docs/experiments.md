# Running the experiments

**Authors:** Anurag Jha, Sandesh Dhital

Everything in [engineering-documentation.md](engineering-documentation.md) is
reproducible from a clean checkout with one command. This page covers how to
run it, what it costs, and what every knob does.

---

## 1. Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Python 3.10+ is required. The dataset ships inside scikit-learn, so nothing is
downloaded at run time.

## 2. The commands

```bash
qml-compare run                       # all four studies, figures, report
qml-compare run --quick               # ~30 s smoke test, numbers not quotable
qml-compare run --study depth         # one study
qml-compare run --study noise-sweep

qml-compare data                      # dataset summary after preprocessing
qml-compare circuit --depth 3         # draw the circuit, count the gates
qml-compare config                    # the resolved configuration
qml-compare report                    # re-render tables and figures from results.json
```

`python -m qml_compare ...` is equivalent if the console script is not on the
path. Helper wrappers live in `scripts/` (`run_all.ps1`, `run_all.sh`).

## 3. Runtime

Measured on a 16-core laptop CPU, defaults throughout:

| Stage | Wall clock |
| --- | --- |
| Dataset preparation | < 1 s |
| Classical baselines (both) | ~0.01 s each |
| VQC depth 1 / 2 / 3, ideal | 4.3 / 5.1 / 5.7 s |
| VQC depth 2, noise-aware training | 265 s |
| Noise sweep (inference only, 14 points) | ~40 s |
| Figures and reports | ~8 s |
| **Full `qml-compare run`** | **5 min 20 s** |

Almost all of it is the one noise-aware training run: it is the only stage that
cannot use the cached-state shortcut described in
[architecture.md](architecture.md#4-backends-where-the-performance-came-from).
Drop it with `--study baseline --study depth` if you only need the noiseless
numbers.

## 4. Output

```
results/
├── results.json          full suite: metrics, timings, loss traces, env
├── results.csv           one flat row per evaluated model
├── report.md             the tables reproduced in the documentation
├── config.used.json      the exact configuration that produced the above
└── figures/
    ├── model_comparison.png      accuracy vs. training cost
    ├── depth_tradeoff.png        accuracy and time vs. ansatz depth
    ├── noise_impact.png          the three noise conditions
    ├── noise_sweep.png           accuracy and margin vs. error rate
    ├── loss_curves.png           COBYLA convergence
    ├── confusion_matrices.png    per-model confusion matrices
    └── dataset_projection.png    the PCA projection, coloured by class
```

`results.json` records the versions of Python, NumPy, SciPy, scikit-learn,
Qiskit and Aer alongside the numbers, so any figure can be traced back to the
stack that produced it.

## 5. Configuration

`configs/default.json` holds every default. Point at your own copy with
`--config`, or override individual values on the command line.

### Top level

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | 42 | Seeds the split, the weight initialisation and the simulators |
| `output_dir` | `results` | Where artefacts are written |
| `depths` | `[1, 2, 3]` | Depths visited by the depth study |
| `reference_depth` | 2 | Depth used by the baseline and noise studies |
| `noise_sweep` | `[0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]` | Gate error rates in the sweep |
| `noise_sweep_shots` | `[0, 1024]` | Shot budgets per sweep point (`0` = exact) |

### `data`

| Key | Default | Meaning |
| --- | --- | --- |
| `n_components` | 4 | PCA components kept — also the qubit count |
| `test_size` | 0.2 | Held-out fraction, stratified |
| `feature_range` | `[-1, 1]` | Post-scaling range, used directly as rotation angles |
| `standardize_before_pca` | `true` | Standardise before PCA (see architecture §2) |

### `circuit`

| Key | Default | Meaning |
| --- | --- | --- |
| `feature_map_reps` | 2 | ZZFeatureMap repetitions |
| `feature_map_entanglement` | `full` | Which qubit pairs get ZZ interactions |
| `feature_map_data_map` | `product` | `product` or `qiskit` (see architecture §3) |
| `ansatz_entanglement` | `linear` | CNOT pattern inside RealAmplitudes |
| `depth` | 2 | Ansatz repetitions when no study overrides it |

### `training`

| Key | Default | Meaning |
| --- | --- | --- |
| `optimizer` | `COBYLA` | Any gradient-free `scipy.optimize.minimize` method |
| `maxiter` | 500 | Objective evaluations |
| `initial_step` | 0.5 | COBYLA `rhobeg`, the initial trust radius |
| `tol` | 1e-4 | Convergence tolerance |
| `l2_regularization` | 0.0 | Penalty on the weight norm |

### `noise`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Set per study; the CLI does not need it |
| `one_qubit_error` | 0.01 | Depolarising probability on single-qubit gates |
| `two_qubit_error` | 0.01 | Depolarising probability on two-qubit gates |
| `readout_error` | 0.0 | Symmetric measurement bit-flip probability |
| `shots` | 0 | `0` = exact density-matrix read-out; `>0` samples that many shots |

## 6. Things worth trying

```bash
# Reproduce the encoding failure described in architecture.md §3
qml-compare run --study baseline --config configs/qiskit-datamap.json

# Does depth ever pay off?
qml-compare run --study depth --depths 1 2 3 4 5 6

# Add sampling noise on top of gate noise
qml-compare run --study noise --shots 1024

# How bad does a device have to be before accuracy notices?
qml-compare run --study noise-sweep --shots 256

# Is the result an artefact of one lucky split?
for seed in 1 2 3 4 5; do qml-compare --seed $seed run --study baseline \
    --output-dir results/seed-$seed --no-figures; done
```

## 7. Tests

```bash
pytest              # whole suite, well under a minute
pytest -k backend   # one area
pytest -q --tb=short
```

The tests run on a two-qubit synthetic problem with tiny optimiser budgets:
real circuits and real backends, small numbers. See
[architecture.md §9](architecture.md#9-testing) for what each file guards.

## 8. Troubleshooting

**A noisy run is much slower than the table above.** Check that
`qiskit-aer` is installed (`qml-compare run --study noise -v` logs the backend
it selected). Without it, the noise study cannot run at all.

**Accuracy differs from the report.** Confirm the seed (`qml-compare config`)
and compare `results/config.used.json` against the committed one. COBYLA is
deterministic for a fixed seed and an exact backend; with `--shots > 0` the
objective is stochastic and small run-to-run differences are expected.

**Memory pressure with more components.** Simulation cost grows as `2^n`.
Beyond 12-14 qubits a laptop will start swapping; `n_components` is the knob
that controls it.
