"""SKAB benchmark protocol.

For every file: the first ``train_rows`` rows (anomaly-free by construction of the
testbed experiments) are used to fit the scaler, the detector and the threshold;
the remaining rows are scored. Metrics are computed over all test points of all
files concatenated, as in the official SKAB leaderboard.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .data.skab import FEATURES, list_skab_files, load_skab_file
from .metrics import detection_metrics, roc_auc
from .pipeline import AnomalyPipeline
from .tracking import Tracker

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model: str
    metrics: dict[str, float]
    n_files: int
    n_test_points: int
    prevalence: float
    seconds: float


def run_file(cfg: Config, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit on the head of one file, return ``(y_true, y_pred, scores)`` for its tail."""
    df = load_skab_file(path)
    x = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["anomaly"].to_numpy()
    n_train, window = cfg.data.train_rows, cfg.data.window
    pipe = AnomalyPipeline(cfg, FEATURES).fit(x[:n_train])
    # prepend the last window-1 training rows so every test point has a full window
    context = x[n_train - window + 1 :]
    scores = pipe.score_windows(context)
    return y[n_train:], scores > pipe.threshold, scores


def run_benchmark(cfg: Config, model: str, tracker: Tracker | None = None) -> BenchmarkResult:
    cfg = copy.deepcopy(cfg)
    cfg.model.name = model
    files = list_skab_files(cfg.data.root, cfg.data.subsets)
    start = time.perf_counter()
    ys, preds, scores = [], [], []
    for path in files:
        y, p, s = run_file(cfg, path)
        ys.append(y)
        preds.append(p)
        scores.append(s)
        log.info(
            "%s %-16s f1=%.3f",
            model,
            f"{path.parent.name}/{path.name}",
            detection_metrics(y, p)["f1"],
        )
    y_all, p_all, s_all = map(np.concatenate, (ys, preds, scores))
    metrics = detection_metrics(y_all, p_all)
    metrics["roc_auc"] = roc_auc(y_all, s_all)
    result = BenchmarkResult(
        model, metrics, len(files), len(y_all), float(y_all.mean()), time.perf_counter() - start
    )
    if tracker is not None:
        with tracker.run(f"benchmark-{model}"):
            tracker.log_params(cfg.to_dict())
            tracker.log_metrics(metrics)
    return result


def write_report(results: list[BenchmarkResult], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda r: -r.metrics["f1"])
    lines = [
        "| Model | F1 | Precision | Recall | FAR % | MAR % | ROC-AUC | Runtime (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r.metrics
        lines.append(
            f"| {r.model} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['far']:.2f} | {m['mar']:.2f} | {m['roc_auc']:.3f} | {r.seconds:.0f} |"
        )
    # Reference: alarm on every point. With a high anomaly ratio in the test split this
    # already yields a non-trivial F1, so F1 must always be read together with FAR.
    prev = rows[0].prevalence if rows else 0.0
    lines.append(
        f"| *always alarm (reference)* | {2 * prev / (1 + prev):.3f} | {prev:.3f} | 1.000 | "
        f"100.00 | 0.00 | 0.500 | – |"
    )
    table = "\n".join(lines) + "\n"
    (out_dir / "results.md").write_text(table)
    (out_dir / "results.json").write_text(
        json.dumps([r.__dict__ for r in rows], indent=2, default=float)
    )
    return out_dir / "results.md"
