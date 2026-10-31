"""Replay a SKAB recording against the scoring API as if it were a live sensor stream.

    python scripts/replay.py data/skab/valve1/0.csv --url http://localhost:8000 --block 30

Each request carries the newest ``block`` readings plus ``window - 1`` readings of
context, so every reading is scored exactly once.
"""

from __future__ import annotations

import argparse
import time

import requests

from iot_anomaly.data.skab import FEATURES, load_skab_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--device-id", default="skab-testbed")
    ap.add_argument("--block", type=int, default=30, help="new readings per request")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    args = ap.parse_args()

    window = requests.get(f"{args.url}/model", timeout=10).json()["window"]
    df = load_skab_file(args.csv)
    x, labels = df[FEATURES].to_numpy(), df["anomaly"].to_numpy()
    times = [t.isoformat() for t in df.index]

    tp = fp = fn = 0
    for end in range(window, len(df) + 1, args.block):
        start = max(0, end - args.block - window + 1)
        body = {
            "device_id": args.device_id,
            "timestamps": times[start:end],
            "readings": x[start:end].tolist(),
        }
        resp = requests.post(f"{args.url}/score", json=body, timeout=30)
        resp.raise_for_status()
        out = resp.json()
        new_points = out["points"][-min(args.block, end - window + 1) :]
        for p in new_points:
            i = start + p["index"]
            truth = bool(labels[i])
            tp += p["is_anomaly"] and truth
            fp += p["is_anomaly"] and not truth
            fn += (not p["is_anomaly"]) and truth
        alarms = sum(p["is_anomaly"] for p in new_points)
        print(
            f"{times[end - 1]}  scored={len(new_points):3d}  alarms={alarms:3d}  "
            f"latency={out['latency_ms']:.1f} ms"
        )
        time.sleep(args.delay)
    print(f"\ntrue alarms={tp}  false alarms={fp}  missed={fn}")


if __name__ == "__main__":
    main()
