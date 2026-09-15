# Architecture

**Authors:** Anurag Jha, Sandesh Dhital

This document explains how QML-Compare is put together and, more usefully, why
each piece is the shape it is. The experiment results live in
[engineering-documentation.md](engineering-documentation.md); this is the
engineering behind them.

---

## 1. Module map

```
src/qml_compare/
├── config.py         frozen dataclasses; one JSON file reproduces a run
├── data.py           Breast Cancer -> standardize -> PCA(4) -> MinMax[-1,1]
├── circuits.py       ZZFeatureMap encoder + RealAmplitudes ansatz
├── backends.py       StatevectorBackend (exact) / AerBackend (noisy)
├── vqc.py            the classifier: parity read-out, cross-entropy, COBYLA
├── classical.py      Logistic Regression and SVM baselines
├── metrics.py        metric computation + the ModelResult record
├── experiments.py    the four studies and the suite runner
├── reporting.py      results.json / results.csv / report.md
├── plotting.py       figures
├── cli.py            argparse front end
└── logging_utils.py  one stream handler, configured once
```

The dependency graph is strictly one-directional:

```
config -> data -> circuits -> backends -> vqc -> experiments -> reporting/plotting -> cli
```

Nothing lower in that chain imports anything higher. That is what makes it
possible to use the classifier on its own:

```python
from qml_compare import VariationalQuantumClassifier

model = VariationalQuantumClassifier(depth=2).fit(x_train, y_train)
model.score(x_test, y_test)
```

## 2. The data pipeline

Breast Cancer Wisconsin has 30 features. One qubit per feature would need a
2^30-amplitude statevector, so the pipeline compresses to four principal
components — four qubits, a 16-amplitude state.

Three decisions are worth naming:

**Standardise before PCA.** PCA maximises variance, and the raw features span
several orders of magnitude (`mean area` ~ 650, `mean smoothness` ~ 0.1).
Without standardisation the first component is essentially "area" and the
projection throws away the rest.

**Scale to `[-1, 1]`, not `[0, 1]`.** The values become rotation angles.
A symmetric range puts the data symmetrically around the Bloch-sphere equator
instead of biasing every sample towards one hemisphere.

**Fit on the training split only.** The split happens first; `StandardScaler`,
`PCA` and `MinMaxScaler` are all fitted on training rows and merely applied to
test rows. Fitting the projection on the full dataset — which the original
specification implies — leaks test-set structure into the model and inflates
the reported accuracy. Test features are clipped back into `[-1, 1]`
afterwards, since a held-out point may legitimately fall outside the training
range. `tests/test_data.py::test_preprocessor_is_fitted_on_training_data_only`
pins this down by rebuilding the split and refitting from scratch.

## 3. The circuit

```
|0>^4 ── ZZFeatureMap(x, reps=2, full) ── RealAmplitudes(theta, reps=d) ── measure
```

The feature map depends only on the sample; the ansatz holds all `4 * (d + 1)`
trainable weights. Keeping them as two separate `QuantumCircuit` objects inside
one `CircuitBundle` is not cosmetic — it is what enables the caching in §4.

### The data map is the single most important tuning choice

Qiskit's `ZZFeatureMap` encodes an interaction between features `i` and `j` as
a phase `2 * prod(pi - x_k)`. With features in `[-1, 1]` that puts every
pairwise phase somewhere between roughly 9 and 34 radians. The encoded state
then oscillates several times across the data range, so two nearby samples land
in unrelated corners of Hilbert space and the optimiser has no gradient to
follow.

Replacing it with the plain product `prod(x_k)` keeps every interaction phase
inside `[-2, 2]` radians. Measured on the held-out split, all else equal:

| Data map | Test accuracy |
| --- | --- |
| Qiskit default `prod(pi - x_k)` | 82.5% |
| Plain product `prod(x_k)` | 92.1% |

Both remain available through `circuit.feature_map_data_map`
(`"product"` / `"qiskit"`), because reproducing the failure is part of the
result.

### Read-out

The parity of the measured bit-string decides the class: an even-parity string
votes 0 and an odd-parity one votes 1, so
`P(y = 1 | x) = probabilities @ parity` and equivalently
`P(y = 1) = (1 - <Z^⊗n>) / 2`. This is the Pauli-Z expectation value the
specification calls for, expressed in a form that also works when the backend
can only return counts.

Reading a probability rather than a raw expectation value is what makes binary
cross-entropy available as the loss. Cross-entropy punishes a confident wrong
answer far harder than a squared error does, which matters when noise is
compressing every prediction towards 0.5.

Note what the read-out does *not* have: no trainable bias, no output scaling,
no classical post-processing layer. Adding a `sigmoid(w * <Z> + b)` head was
tried and produced slightly *worse* held-out accuracy (92.1% -> 91.2% at depth
2) while making it harder to say which part of the model was doing the work.

## 4. Backends: where the performance came from

A naive implementation evaluates one circuit per sample per optimiser
iteration: 455 x 500 = 227,500 simulations for a single training run. Both
backends avoid that in different ways.

### `StatevectorBackend` — exact, and fast because of the split circuit

The feature map does not depend on `theta`. So the encoded states are computed
**once** and cached:

```
Psi = [ |psi(x_1)>, ..., |psi(x_n)> ]        # (n, 16), computed once
U(theta) = Operator(ansatz(theta))           # (16, 16), once per iteration
probabilities = |Psi @ U(theta).T|^2         # one BLAS call for the whole batch
```

Per iteration the cost collapses from 455 circuit simulations to one small
matrix construction and one matrix product. A 500-iteration training run
finishes in about 5 seconds instead of several minutes.

### `AerBackend` — noise, at a price

Once a depolarising channel is in the circuit the evolution is no longer
unitary and the cache trick dies. Three things keep it tractable:

1. **`method="density_matrix"`.** With a noise model, Aer's default method
   samples one stochastic *trajectory per shot*. The density-matrix method
   evolves the state once and reads or samples the result — measured at
   ~8x faster per iteration for this circuit.
2. **One batched job per iteration.** The circuit is transpiled once at
   construction; each iteration submits a single `run()` with
   `parameter_binds` covering the whole batch, rather than 455 Python-level
   calls.
3. **`max_parallel_experiments`.** The batch is spread across cores.

Measured per-iteration cost on the reference machine (455 samples, 4 qubits,
depth 2):

| Configuration | Per iteration | 500 iterations |
| --- | --- | --- |
| Default method, 1024 shots | 3.7 s | 31 min |
| Density matrix, 1024 shots, parallel | 2.2 s | 18 min |
| Density matrix, exact read-out, parallel | 0.53 s | 4.4 min |

`shots = 0` therefore means "exact density-matrix probabilities", the
infinite-shot limit. It is the default for the noise study because it isolates
decoherence from sampling noise; `--shots 1024` adds the sampling noise back
when that interaction is the thing under test.

## 5. The training loop

```python
for each COBYLA proposal theta:
    p     = backend.probabilities(x_train, theta) @ parity
    loss  = -mean(y * log p + (1 - y) * log(1 - p)) + l2 * ||theta||^2
```

**COBYLA**, as specified, is a gradient-free trust-region method. That is the
right family here: with shot-based read-out the objective is stochastic, and a
finite-difference gradient would mostly measure sampling noise. The budget is
500 objective evaluations (the original 200 stopped short of convergence; 2000
bought a further 0.4 pp for four times the cost).

**Initialisation** draws weights uniformly from `[-0.1, 0.1]`. Large random
angles start the optimiser in a barren plateau where the objective is flat in
every direction and COBYLA's initial simplex learns nothing.

`TrainingHistory` records the loss and elapsed time at every objective
evaluation, which is what `loss_curves.png` plots and what makes a claim like
"it had converged" checkable rather than asserted.

## 6. Why the experiments are shaped the way they are

**Shared models.** The reference-depth noiseless VQC appears in the baseline
study, the depth sweep, the noise study and the noise sweep. `_VQCCache`
trains it once and hands the same fitted object to all four, so every table in
the report is backed by the same weights — and a full run costs one training
per configuration, not one per appearance.

**The noise study has three arms, not two.** Reporting a single "with noise"
number hides the question that actually matters:

- *Ideal* — the noiseless reference.
- *Noisy inference* — weights trained in simulation, then deployed on a noisy
  device. `VariationalQuantumClassifier.transfer_to()` swaps the backend and
  keeps the weights, so the difference isolates inference noise. This is what
  "run your trained model on hardware" means in practice.
- *Noise-aware training* — the optimiser itself sees the noisy device.

**The noise sweep exists because accuracy is a lagging indicator.**
Depolarising noise contracts every probability towards 0.5, and 0.5 is exactly
the decision threshold — so hard predictions survive long after the model has
stopped being useful. The sweep therefore records the mean margin
`|P(y=1) - 0.5|` next to accuracy, and the two tell visibly different stories.

## 7. Reporting

`ModelResult` is the single record type: one row per evaluated model, carrying
metrics, timings, the confusion matrix, backend/circuit metadata and (for
quantum models) the full loss trace. Everything downstream — CSV, Markdown
tables, every figure — is a projection of a list of those records, so adding a
metric means touching one dataclass.

`results.json` also stores the resolved config, the dataset summary and the
versions of Python, NumPy, SciPy, scikit-learn, Qiskit and Aer, so a number in
the report can always be traced back to the stack that produced it.

## 8. Deliberate deviations from the original specification

| Specification | What was built | Why |
| --- | --- | --- |
| PCA/scaling before the split | Fitted on the training split only | The original order leaks test statistics into the model |
| `qiskit-machine-learning`'s `VQC` | Training loop implemented directly on Qiskit primitives | Needed control over the read-out, the loss and per-iteration history; also removes a dependency that lags Qiskit releases |
| Default `ZZFeatureMap` data map | Plain-product data map | The default puts interaction phases at 9-34 rad for `[-1, 1]` inputs and costs ~10 pp of accuracy |
| 1% depolarising noise, single "with noise" number | Three-arm noise study plus a sweep | A single number conflates deployment noise with training noise, and hides that the margin collapses long before accuracy does |
| Shot-based noisy simulation | Exact density-matrix read-out by default, shots opt-in | Separates decoherence from sampling noise, and is ~4x faster |

## 9. Testing

`pytest` runs the whole suite in well under a minute by using a two-qubit
synthetic problem and tiny optimiser budgets — real circuits, real backends,
real training loop, small numbers. The tests that carry the most weight:

- `test_data.py` — no leakage, correct ranges, deterministic splits.
- `test_backends.py` — Aer and the statevector path agree to 1e-8 without
  noise, shot sampling converges to the exact distribution, noise moves the
  distribution towards uniform, and the cache is keyed on data rather than
  object identity.
- `test_vqc.py` — the loss actually decreases, the same seed reproduces the
  same weights, and `transfer_to` preserves both weights and training
  attribution.
- `test_experiments.py` — the shared-model cache really does hand the same
  fitted object to every study.
