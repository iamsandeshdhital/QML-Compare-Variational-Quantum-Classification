# QML-Compare

**Variational quantum classification, benchmarked honestly against classical machine learning.**

Authors: **Anurag Jha** and **Sandesh Dhital**

A Variational Quantum Classifier (VQC) built on Qiskit, trained on the Breast
Cancer Wisconsin dataset, and measured against Logistic Regression and an RBF
SVM on identical data — plus two studies that ask what actually limits a
variational model today: circuit depth, and hardware noise.

---

## The result in one table

| Model | Test accuracy | F1 | ROC-AUC | Training time |
| --- | --- | --- | --- | --- |
| Logistic Regression | **95.6%** | 0.966 | 0.993 | 0.01 s |
| SVM (RBF kernel) | **95.6%** | 0.966 | 0.992 | 0.01 s |
| VQC (depth 2, noiseless) | 93.9% | 0.953 | 0.979 | 5.1 s |

The VQC is competitive — two extra misclassifications out of 114 — and about
**700x** slower to train. On this kind of tabular data, classical methods win
decisively.

Three findings from getting there are more interesting than the headline:

**1. The default encoding was crippling the model.** Qiskit's `ZZFeatureMap`
maps feature interactions to a phase `2·∏(π − xₖ)`. With features scaled to
`[-1, 1]` that puts every pairwise phase between 9 and 34 radians — the
encoded state oscillates across the data range and the optimiser has nothing
to follow. Swapping in a plain product `∏xₖ` keeps phases inside `[-2, 2]`:

| Data map | Test accuracy |
| --- | --- |
| Qiskit default `∏(π − xₖ)` | 82.5% |
| Plain product `∏xₖ` | **92.1%** |

**2. Depth saturates immediately.** Depth 1 → 2 buys 3.5 points. Depth 2 → 3
*loses* 1.8 while costing more time. Expressibility is not accuracy.

| Depth | Weights | Test accuracy |
| --- | --- | --- |
| 1 | 8 | 90.4% |
| 2 | 12 | **93.9%** |
| 3 | 16 | 92.1% |

**3. Noise destroys confidence long before it destroys accuracy.** A
depolarising channel contracts every probability towards 0.5 — which is
exactly the decision threshold, so predictions survive while the signal dies.
At 10% gate error the model still scores 86.8% with exact read-out, but its
mean margin `|P(y=1) − 0.5|` has fallen from 0.224 to 0.003. Measure the same
weights with a realistic 1024-shot budget and accuracy collapses to **59.6%**,
because the margin is now five times below the sampling floor.

| Gate error | Exact read-out | 1024 shots | Mean margin |
| --- | --- | --- | --- |
| 0% | 93.9% | 93.0% | 0.224 |
| 1% | 92.1% | 92.1% | 0.143 |
| 5% | 89.5% | 85.1% | 0.025 |
| 10% | 86.8% | **59.6%** | 0.003 |
| 20% | 78.1% | 54.4% | 0.000 |

A VQC does not fail because noise makes it wrong. It fails because noise
shrinks the read-out signal below what a realistic shot budget can resolve.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"

qml-compare run                 # all four studies -> results/  (~5 min)
qml-compare run --quick         # ~30 s smoke test
pytest                          # the full test suite, under a minute
```

Other entry points:

```bash
qml-compare data                # dataset summary after preprocessing
qml-compare circuit --depth 3   # draw the circuit and count its gates
qml-compare config              # the resolved configuration
qml-compare report              # re-render tables and figures from results.json
qml-compare run --study noise-sweep --shots 256
```

Use the classifier on its own — it follows the scikit-learn estimator
contract:

```python
from qml_compare import VariationalQuantumClassifier, load_dataset

data = load_dataset()
model = VariationalQuantumClassifier(depth=2).fit(data.x_train, data.y_train)
print(model.score(data.x_test, data.y_test))
```

## How it works

```
Breast Cancer (30 features)
        │  standardise → PCA(4) → MinMax[-1, 1]        fitted on train only
        v
   4 features ──> |0⟩⊗4 ─ ZZFeatureMap(x) ─ RealAmplitudes(θ) ─ measure
                                                                    │
                          parity read-out  P(y=1) = (1 − ⟨Z⊗4⟩)/2 ◄─┘
                                    │
                     binary cross-entropy ──> COBYLA ──> new θ
```

Four qubits, one per principal component. The feature map encodes the sample;
the ansatz holds the `4·(d+1)` trainable weights. Parity of the measured
bit-string gives `P(y = 1 | x)` directly, which is what makes cross-entropy
available as the loss.

**Why it trains in seconds.** A naive loop simulates one circuit per sample
per iteration — 227,500 simulations per training run. Since the feature map
does not depend on the weights, the encoded states are computed **once** and
cached; each iteration then applies the ansatz as a single 16×16 unitary to
the whole batch in one matrix product. Noisy runs cannot use that shortcut, so
they go through Aer's density-matrix method with one batched job per iteration
— still ~8x faster than the default trajectory sampling.

## Layout

```
src/qml_compare/     the package (config, data, circuits, backends, vqc,
                     classical, metrics, experiments, reporting, plotting, cli)
tests/               98 tests on a 2-qubit synthetic problem — fast, but real
                     circuits, real backends, real training loop
docs/
├── engineering-documentation.md   the full study: method, results, limits
├── architecture.md                design decisions and why
└── experiments.md                 reproduction guide and config reference
configs/             default.json plus the deliberately-broken data map
results/             committed output: report.md, results.{json,csv}, figures/
scripts/             run_all.{sh,ps1}, quick_check.sh
```

## Reproducibility

Every run writes `results/config.used.json` alongside its numbers, and
`results.json` records the versions of Python, NumPy, SciPy, scikit-learn,
Qiskit and Aer. With a fixed seed and exact read-out the pipeline is fully
deterministic; with `--shots > 0` the objective is stochastic and small
run-to-run differences are expected.

The committed results come from seed 42 on `configs/default.json`.

## Limitations

Worth stating plainly, and covered at length in
[the documentation](docs/engineering-documentation.md#11-limitations):
simulation caps the model at 4 qubits, which discards 20% of the dataset's
variance before training starts; the dataset is small and nearly linearly
separable, so 114 test samples make one misclassification worth 0.9 points;
all headline numbers come from a single split at one seed; and the noise model
is a uniform depolarising channel, which is a crude stand-in for a real device.

## License

MIT — see [LICENSE](LICENSE).
