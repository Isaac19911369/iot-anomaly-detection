"""Optional MLflow tracking. Becomes a no-op when disabled or mlflow is not installed."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .config import TrackingConfig

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.enabled = cfg.enabled
        self._mlflow = None
        if self.enabled:
            try:
                import mlflow
            except ImportError:
                log.warning("mlflow not installed; tracking disabled (pip install '.[tracking]')")
                self.enabled = False
                return
            mlflow.set_tracking_uri(cfg.tracking_uri)
            mlflow.set_experiment(cfg.experiment)
            self._mlflow = mlflow

    @contextlib.contextmanager
    def run(self, name: str, nested: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with self._mlflow.start_run(run_name=name, nested=nested):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params(_flatten(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifacts(self, path: str | Path) -> None:
        if self.enabled:
            self._mlflow.log_artifacts(str(path))


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
