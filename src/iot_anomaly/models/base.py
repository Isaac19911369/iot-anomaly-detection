"""Common detector interface: fit on normal windows, score unseen windows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Detector(ABC):
    """Unsupervised detector operating on windows of shape ``(N, window, features)``.

    Higher scores mean more anomalous.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, windows: np.ndarray) -> Detector: ...

    @abstractmethod
    def score(self, windows: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @abstractmethod
    def load(self, directory: Path) -> Detector: ...
