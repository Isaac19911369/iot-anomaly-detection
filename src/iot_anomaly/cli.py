"""Command-line interface: download, benchmark, train, serve."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .benchmark import run_benchmark, write_report
from .config import Config
from .data.skab import FEATURES, download_skab, load_skab_file
from .models import MODEL_NAMES
from .pipeline import AnomalyPipeline
from .tracking import Tracker


def _cmd_download(args: argparse.Namespace, cfg: Config) -> None:
    root = download_skab(args.root or cfg.data.root, force=args.force)
    print(f"SKAB data available in {root}")


def _cmd_benchmark(args: argparse.Namespace, cfg: Config) -> None:
    tracker = Tracker(cfg.tracking)
    results = []
    for name in args.models:
        r = run_benchmark(cfg, name, tracker)
        print(
            f"{name:8s} F1={r.metrics['f1']:.3f} FAR={r.metrics['far']:.2f}% "
            f"MAR={r.metrics['mar']:.2f}% AUC={r.metrics['roc_auc']:.3f} ({r.seconds:.0f}s)"
        )
        results.append(r)
    report = write_report(results, Path(cfg.output_dir) / "benchmark")
    print(f"\nReport written to {report}\n")
    print(report.read_text())


def _cmd_train(args: argparse.Namespace, cfg: Config) -> None:
    if args.model:
        cfg.model.name = args.model
    path = Path(args.csv or Path(cfg.data.root) / "anomaly-free" / "anomaly-free.csv")
    df = load_skab_file(path)
    x = df[FEATURES].to_numpy(dtype=np.float64)
    if args.rows:
        x = x[: args.rows]
    tracker = Tracker(cfg.tracking)
    with tracker.run(f"train-{cfg.model.name}"):
        pipe = AnomalyPipeline(cfg, FEATURES).fit(x)
        out = pipe.save(args.out or Path(cfg.output_dir) / "model")
        tracker.log_params(cfg.to_dict())
        tracker.log_metrics({"threshold": pipe.threshold, "train_rows": len(x)})
        tracker.log_artifacts(out)
    print(
        f"Trained {cfg.model.name} on {len(x)} rows; threshold={pipe.threshold:.5f}; saved to {out}"
    )


def _cmd_serve(args: argparse.Namespace, cfg: Config) -> None:
    import os

    import uvicorn

    os.environ.setdefault("MODEL_DIR", str(args.model_dir or Path(cfg.output_dir) / "model"))
    uvicorn.run("iot_anomaly.serving.api:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iot-anomaly", description=__doc__)
    p.add_argument("-c", "--config", default=None, help="YAML config file")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download the SKAB dataset")
    d.add_argument("--root", default=None)
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=_cmd_download)

    b = sub.add_parser("benchmark", help="run the SKAB per-file benchmark")
    b.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    b.set_defaults(func=_cmd_benchmark)

    t = sub.add_parser("train", help="train a serving model on normal data")
    t.add_argument("--model", choices=MODEL_NAMES, default=None)
    t.add_argument("--csv", default=None, help="SKAB-format CSV (default: anomaly-free.csv)")
    t.add_argument("--rows", type=int, default=None, help="use only the first N rows (normal)")
    t.add_argument("--out", default=None)
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("serve", help="start the scoring API")
    s.add_argument("--model-dir", default=None)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args, Config.load(args.config))


if __name__ == "__main__":
    main()
