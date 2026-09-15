"""Experiment configuration.

Everything that changes the numbers in ``results/`` lives here, so a run can be
reproduced from a single JSON file plus the recorded git revision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.json"


@dataclass(frozen=True)
class DataConfig:
    """Dataset preparation settings."""

    n_components: int = 4
    test_size: float = 0.2
    feature_range: tuple[float, float] = (-1.0, 1.0)
    standardize_before_pca: bool = True


@dataclass(frozen=True)
class CircuitConfig:
    """Parameterised quantum circuit settings."""

    feature_map_reps: int = 2
    feature_map_entanglement: str = "full"
    feature_map_data_map: str = "product"
    ansatz_entanglement: str = "linear"
    depth: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Classical optimisation loop settings."""

    optimizer: str = "COBYLA"
    maxiter: int = 500
    initial_step: float = 0.5
    tol: float = 1e-4
    l2_regularization: float = 0.0


@dataclass(frozen=True)
class NoiseConfig:
    """Depolarising noise model settings for the Aer backend."""

    enabled: bool = False
    one_qubit_error: float = 0.01
    two_qubit_error: float = 0.01
    readout_error: float = 0.0
    #: 0 means exact density-matrix probabilities (the infinite-shot limit);
    #: any positive value samples that many shots and adds sampling noise.
    shots: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration object passed through the whole pipeline."""

    seed: int = 42
    output_dir: Path = Path("results")
    depths: tuple[int, ...] = (1, 2, 3)
    reference_depth: int = 2
    #: Gate error rates evaluated by the noise sweep (0.0 is the ideal point).
    noise_sweep: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    #: Shot budgets evaluated at each sweep point; 0 means exact read-out.
    noise_sweep_shots: tuple[int, ...] = (0, 1024)
    data: DataConfig = field(default_factory=DataConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # ------------------------------------------------------------------ IO --
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        """Build a config from a (possibly partial) mapping."""
        raw = dict(raw)
        nested = {
            "data": DataConfig,
            "circuit": CircuitConfig,
            "training": TrainingConfig,
            "noise": NoiseConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, factory in nested.items():
            section = dict(raw.pop(name, {}) or {})
            if "feature_range" in section:
                section["feature_range"] = tuple(section["feature_range"])
            _reject_unknown(factory, section, name)
            kwargs[name] = factory(**section)

        if "output_dir" in raw:
            raw["output_dir"] = Path(raw["output_dir"])
        if "depths" in raw:
            raw["depths"] = tuple(int(d) for d in raw["depths"])
        if "noise_sweep" in raw:
            raw["noise_sweep"] = tuple(float(v) for v in raw["noise_sweep"])
        if "noise_sweep_shots" in raw:
            raw["noise_sweep_shots"] = tuple(int(v) for v in raw["noise_sweep_shots"])
        _reject_unknown(cls, raw, "config")
        return cls(**raw, **kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load a config from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def default(cls) -> "ExperimentConfig":
        """Load ``configs/default.json`` when present, else the dataclass defaults."""
        if DEFAULT_CONFIG_PATH.exists():
            return cls.from_json(DEFAULT_CONFIG_PATH)
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the configuration."""
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["depths"] = list(self.depths)
        payload["noise_sweep"] = list(self.noise_sweep)
        payload["noise_sweep_shots"] = list(self.noise_sweep_shots)
        payload["data"]["feature_range"] = list(self.data.feature_range)
        return payload

    def to_json(self, path: str | Path) -> Path:
        """Write the configuration to ``path`` and return it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    # ------------------------------------------------------------- helpers --
    def replace(self, **changes: Any) -> "ExperimentConfig":
        """Return a copy with top-level or ``section.field`` overrides applied.

        ``cfg.replace(**{"circuit.depth": 3})`` rewrites the nested dataclass,
        which keeps the sweep code free of manual reconstruction noise.
        """
        payload = self.to_dict()
        for key, value in changes.items():
            if "." in key:
                section, attr = key.split(".", 1)
                payload[section][attr] = value
            else:
                payload[key] = value
        return ExperimentConfig.from_dict(payload)


def _reject_unknown(factory: Any, values: Mapping[str, Any], where: str) -> None:
    known = {f.name for f in fields(factory)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(unknown)}")
