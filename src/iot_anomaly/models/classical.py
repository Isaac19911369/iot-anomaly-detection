"""Classical baselines: PCA reconstruction (SPE / Q-statistic) and Isolation Forest.

Both operate on the most recent reading of each window (point-wise), which is how
these methods are typically used in process monitoring. Flattening whole windows
gives hundreds of dimensions for a few hundred training samples and overfits.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .base import Detector


class PCADetector(Detector):
    """Squared prediction error (Q-statistic) of a PCA reconstruction of the current reading."""

    name = "pca"

    def __init__(self, n_components: float = 0.95, seed: int = 42):
        self.model = PCA(n_components=n_components, random_state=seed)

    def fit(self, windows: np.ndarray) -> PCADetector:
        self.model.fit(windows[:, -1, :])
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        flat = windows[:, -1, :]
        recon = self.model.inverse_transform(self.model.transform(flat))
        return np.mean((flat - recon) ** 2, axis=1)

    def save(self, directory: Path) -> None:
        joblib.dump(self.model, Path(directory) / "model.joblib")

    def load(self, directory: Path) -> PCADetector:
        self.model = joblib.load(Path(directory) / "model.joblib")
        return self


class IsolationForestDetector(Detector):
    """Isolation Forest on the current reading."""

    name = "iforest"

    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)

    def fit(self, windows: np.ndarray) -> IsolationForestDetector:
        self.model.fit(windows[:, -1, :])
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(windows[:, -1, :])

    def save(self, directory: Path) -> None:
        joblib.dump(self.model, Path(directory) / "model.joblib")

    def load(self, directory: Path) -> IsolationForestDetector:
        self.model = joblib.load(Path(directory) / "model.joblib")
        return self
