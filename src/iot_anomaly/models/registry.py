"""Build detectors by name."""

from __future__ import annotations

from pathlib import Path

from ..config import ModelConfig, TrainingConfig
from .autoencoders import TorchAutoencoderDetector
from .base import Detector
from .classical import IsolationForestDetector, PCADetector

MODEL_NAMES = ("pca", "iforest", "lstm_ae", "conv_ae")


def build_detector(model_cfg: ModelConfig, train_cfg: TrainingConfig, n_features: int) -> Detector:
    name = model_cfg.name
    if name == "pca":
        return PCADetector(model_cfg.n_components, seed=train_cfg.seed)
    if name == "iforest":
        return IsolationForestDetector(model_cfg.n_estimators, seed=train_cfg.seed)
    if name in ("lstm_ae", "conv_ae"):
        return TorchAutoencoderDetector(name, n_features, model_cfg, train_cfg)
    raise ValueError(f"Unknown model '{name}'. Choose one of {MODEL_NAMES}.")


def load_detector(
    model_cfg: ModelConfig, train_cfg: TrainingConfig, n_features: int, directory: Path
) -> Detector:
    return build_detector(model_cfg, train_cfg, n_features).load(directory)
