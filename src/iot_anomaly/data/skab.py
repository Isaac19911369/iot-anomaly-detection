"""Loader for the Skoltech Anomaly Benchmark (SKAB).

SKAB contains multivariate sensor recordings (accelerometers, current, pressure,
temperature, voltage, flow rate) from a water-circulation testbed, with labelled
anomalies caused by valve manipulation, cavitation and other faults.

Data is downloaded at runtime from the official repository and is not
redistributed here: https://github.com/waico/SKAB
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data"
TREE_URL = "https://api.github.com/repos/waico/SKAB/git/trees/master?recursive=1"

FEATURES = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

# Files of the benchmark split, used as a fallback when the GitHub API is unreachable.
_KNOWN_FILES = (
    [f"valve1/{i}.csv" for i in range(16)]
    + [f"valve2/{i}.csv" for i in range(4)]
    + [f"other/{i}.csv" for i in range(1, 15)]
    + ["anomaly-free/anomaly-free.csv"]
)


def _remote_files() -> list[str]:
    try:
        resp = requests.get(TREE_URL, timeout=20)
        resp.raise_for_status()
        return [
            t["path"].removeprefix("data/")
            for t in resp.json()["tree"]
            if t["path"].startswith("data/") and t["path"].endswith(".csv")
        ]
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("Could not list SKAB files via GitHub API (%s); using built-in list", exc)
        return list(_KNOWN_FILES)


def download_skab(root: str | Path = "data/skab", force: bool = False) -> Path:
    """Download all SKAB CSV files into ``root`` (skips files already present)."""
    root = Path(root)
    for rel in _remote_files():
        dest = root / rel
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(f"{BASE_URL}/{rel}", timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        log.info("downloaded %s", rel)
    return root


def list_skab_files(root: str | Path, subsets: list[str]) -> list[Path]:
    """Benchmark files for the given subsets, in a stable numeric order."""
    root = Path(root)
    files: list[Path] = []
    for subset in subsets:
        found = sorted((root / subset).glob("*.csv"), key=lambda p: (len(p.stem), p.stem))
        if not found:
            raise FileNotFoundError(
                f"No SKAB files in {root / subset}. Run `iot-anomaly download` first."
            )
        files.extend(found)
    return files


def load_skab_file(path: str | Path) -> pd.DataFrame:
    """Load one SKAB file indexed by timestamp, with ``anomaly`` labels when present."""
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"]).set_index("datetime")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing feature columns {missing}")
    if "anomaly" not in df.columns:
        df["anomaly"] = 0
    df["anomaly"] = df["anomaly"].fillna(0).astype(int)
    return df[FEATURES + ["anomaly"]]
