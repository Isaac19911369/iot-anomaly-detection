"""Synthetic multivariate sensor data with injected anomalies (tests and demos)."""

from __future__ import annotations

import numpy as np


def make_synthetic_series(
    n: int = 2000,
    n_features: int = 4,
    anomaly_ratio: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated periodic signals plus noise, with spike and level-shift anomalies.

    Returns ``(x, y)`` where ``x`` has shape ``(n, n_features)`` and ``y`` is a 0/1 label.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / 50)
    mix = rng.normal(size=(1, n_features))
    phase = rng.uniform(0, 2 * np.pi, size=n_features)
    x = base[:, None] * mix + 0.3 * np.sin(2 * np.pi * t[:, None] / 13 + phase)
    x += rng.normal(scale=0.05, size=x.shape)
    y = np.zeros(n, dtype=int)

    n_events = max(1, int(n * anomaly_ratio / 10))
    starts = rng.choice(np.arange(n // 2, n - 20), size=n_events, replace=False)
    for s in starts:
        length = int(rng.integers(5, 15))
        feat = rng.integers(0, n_features)
        x[s : s + length, feat] += rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
        y[s : s + length] = 1
    return x.astype(np.float32), y
