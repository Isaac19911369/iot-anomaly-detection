"""End-to-end pipeline: scaling -> windowing -> detector -> threshold, with persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, ModelConfig, ThresholdConfig, TrainingConfig
from .data.windowing import StandardScaler, make_windows, window_scores_to_points
from .models import Detector, build_detector, load_detector


def calibrate_threshold(train_scores: np.ndarray, cfg: ThresholdConfig) -> float:
    """``median + scale * (quantile - median)`` of the scores on normal training data.

    For reconstruction errors (median close to 0) this reduces to the classic
    ``scale * quantile`` upper control limit; subtracting the median keeps the rule
    meaningful for detectors whose scores have an offset (e.g. Isolation Forest).
    """
    med = float(np.median(train_scores))
    return med + cfg.scale * (float(np.quantile(train_scores, cfg.quantile)) - med)


class AnomalyPipeline:
    def __init__(self, config: Config, feature_names: list[str]):
        self.config = config
        self.feature_names = list(feature_names)
        self.window = config.data.window
        self.scaler = StandardScaler()
        self.detector: Detector | None = None
        self.threshold: float | None = None

    # ------------------------------------------------------------------ training
    def fit(self, x_train: np.ndarray) -> AnomalyPipeline:
        """Fit on (assumed) normal data and calibrate the alarm threshold on it."""
        x = self.scaler.fit_transform(np.asarray(x_train, dtype=np.float64))
        windows = make_windows(x, self.window)
        self.detector = build_detector(
            self.config.model, self.config.training, n_features=x.shape[1]
        ).fit(windows)
        train_scores = self.detector.score(windows)
        self.threshold = calibrate_threshold(train_scores, self.config.threshold)
        return self

    # ------------------------------------------------------------------ inference
    def score_windows(self, x: np.ndarray) -> np.ndarray:
        if self.detector is None:
            raise RuntimeError("Pipeline is not fitted")
        x = self.scaler.transform(np.asarray(x, dtype=np.float64))
        return self.detector.score(make_windows(x, self.window))

    def score_points(self, x: np.ndarray) -> np.ndarray:
        """Per-timestep scores; the first ``window - 1`` points are ``nan``."""
        return window_scores_to_points(self.score_windows(x), self.window)

    def predict_points(self, x: np.ndarray) -> np.ndarray:
        scores = self.score_points(x)
        return np.nan_to_num(scores, nan=-np.inf) > self.threshold

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        assert self.detector is not None, "fit before saving"
        self.detector.save(directory)
        meta = {
            "version": __version__,
            "model": self.config.model.name,
            "window": self.window,
            "threshold": self.threshold,
            "features": self.feature_names,
            "scaler": self.scaler.to_dict(),
            "config": self.config.to_dict(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> AnomalyPipeline:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        config = Config.from_dict(meta["config"])
        pipe = cls(config, meta["features"])
        pipe.scaler = StandardScaler.from_dict(meta["scaler"])
        pipe.threshold = float(meta["threshold"])
        pipe.detector = load_detector(
            ModelConfig(**asdict(config.model)),
            TrainingConfig(**{**asdict(config.training), "device": "cpu"}),
            n_features=len(meta["features"]),
            directory=directory,
        )
        return pipe
