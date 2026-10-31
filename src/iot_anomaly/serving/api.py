"""FastAPI scoring service.

POST a block of consecutive sensor readings; every reading that completes a full
window receives an anomaly score and a flag. Clients streaming data should send
the last ``window - 1`` readings of the previous block as context.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..pipeline import AnomalyPipeline
from .storage import ScoreStore


class ScoreRequest(BaseModel):
    device_id: str = Field("default", examples=["pump-01"])
    timestamps: list[datetime] | None = Field(
        None, description="Optional, one per reading; enables persistence to TimescaleDB"
    )
    readings: list[list[float]] = Field(
        ..., description="Consecutive readings, each a list of feature values in model order"
    )


class ScoredPoint(BaseModel):
    index: int
    timestamp: datetime | None = None
    score: float
    is_anomaly: bool


class ScoreResponse(BaseModel):
    model: str
    threshold: float
    window: int
    points: list[ScoredPoint]
    n_anomalies: int
    latency_ms: float


@lru_cache(maxsize=1)
def get_pipeline() -> AnomalyPipeline:
    return AnomalyPipeline.load(os.environ.get("MODEL_DIR", "artifacts/model"))


@lru_cache(maxsize=1)
def get_store() -> ScoreStore:
    return ScoreStore(os.environ.get("DATABASE_URL"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_pipeline()  # fail fast if the model bundle is missing
    yield


app = FastAPI(title="IoT Anomaly Detection API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict:
    pipe = get_pipeline()
    return {
        "model": pipe.config.model.name,
        "window": pipe.window,
        "threshold": pipe.threshold,
        "features": pipe.feature_names,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    pipe = get_pipeline()
    x = np.asarray(req.readings, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(pipe.feature_names):
        raise HTTPException(
            422,
            f"each reading needs {len(pipe.feature_names)} values in order {pipe.feature_names}",
        )
    if len(x) < pipe.window:
        raise HTTPException(422, f"need at least {pipe.window} consecutive readings")
    if req.timestamps is not None and len(req.timestamps) != len(x):
        raise HTTPException(422, "timestamps must have the same length as readings")

    start = time.perf_counter()
    scores = pipe.score_windows(x)
    flags = scores > pipe.threshold
    offset = pipe.window - 1
    ts = req.timestamps[offset:] if req.timestamps else [None] * len(scores)
    points = [
        ScoredPoint(index=offset + i, timestamp=t, score=float(s), is_anomaly=bool(f))
        for i, (t, s, f) in enumerate(zip(ts, scores, flags, strict=True))
    ]
    if req.timestamps:
        get_store().write(req.device_id, pipe.config.model.name, ts, scores, flags)
    return ScoreResponse(
        model=pipe.config.model.name,
        threshold=pipe.threshold,
        window=pipe.window,
        points=points,
        n_anomalies=int(flags.sum()),
        latency_ms=(time.perf_counter() - start) * 1000,
    )
