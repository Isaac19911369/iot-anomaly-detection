"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/skab"
    subsets: list[str] = field(default_factory=lambda: ["valve1", "valve2", "other"])
    train_rows: int = 400  # SKAB protocol: first N rows of each file are anomaly-free
    window: int = 60


@dataclass
class ModelConfig:
    name: str = "conv_ae"  # pca | iforest | lstm_ae | conv_ae
    hidden_dim: int = 32
    latent_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.1
    score_on: str = "last"  # autoencoders: "last" = error of the newest reading, "window" = mean
    n_components: float = 0.95  # PCA: explained variance to keep
    n_estimators: int = 200  # Isolation Forest


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "auto"


@dataclass
class ThresholdConfig:
    quantile: float = 0.99
    scale: float = 1.5  # threshold = median + scale * (quantile - median) of train scores


@dataclass
class TrackingConfig:
    enabled: bool = False
    tracking_uri: str = "mlruns"
    experiment: str = "iot-anomaly-detection"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output_dir: str = "artifacts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Config:
        raw = raw or {}
        kwargs = {}
        for f in fields(cls):
            value = raw.get(f.name)
            if value is None:
                continue
            sub = f.default_factory() if callable(f.default_factory) else None  # type: ignore[misc]
            if sub is not None and hasattr(sub, "__dataclass_fields__"):
                unknown = set(value) - set(sub.__dataclass_fields__)
                if unknown:
                    raise ValueError(f"Unknown keys in '{f.name}': {sorted(unknown)}")
                kwargs[f.name] = type(sub)(**value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))
