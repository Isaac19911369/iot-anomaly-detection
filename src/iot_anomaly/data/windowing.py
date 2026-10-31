"""Sliding windows and scaling for multivariate time series."""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Minimal, serialisable per-feature standardisation."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray) -> StandardScaler:
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted")
        return ((x - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        return {"mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> StandardScaler:
        return cls(np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64))


def make_windows(x: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """``(T, F)`` -> ``(N, window, F)`` with ``N = (T - window) // stride + 1``."""
    if x.ndim != 2:
        raise ValueError(f"expected 2-D array (time, features), got shape {x.shape}")
    if len(x) < window:
        raise ValueError(f"series of length {len(x)} is shorter than window {window}")
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)  # (N, F, window)
    return np.ascontiguousarray(view[::stride].transpose(0, 2, 1))


def window_scores_to_points(scores: np.ndarray, window: int) -> np.ndarray:
    """Assign each window's score to its last timestep (causal, streaming-friendly).

    The first ``window - 1`` points have no full window; they get ``nan``.
    """
    return np.concatenate([np.full(window - 1, np.nan), scores])
