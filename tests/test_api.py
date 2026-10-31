import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from iot_anomaly.pipeline import AnomalyPipeline

FEATURES = ["f0", "f1", "f2", "f3"]


@pytest.fixture
def client(series, fast_config, tmp_path, monkeypatch):
    x, _ = series
    fast_config.model.name = "pca"
    AnomalyPipeline(fast_config, FEATURES).fit(x[:600]).save(tmp_path)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    api = importlib.import_module("iot_anomaly.serving.api")
    api.get_pipeline.cache_clear()
    api.get_store.cache_clear()
    with TestClient(api.app) as c:
        yield c


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model").json()
    assert info["features"] == FEATURES and info["window"] == 20


def test_score_flags_injected_spike(client, series):
    x, _ = series
    block = x[100:160].copy()
    block[-1, 0] += 25.0  # obvious spike on the newest reading
    resp = client.post("/score", json={"device_id": "pump-01", "readings": block.tolist()})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["points"]) == len(block) - body["window"] + 1
    assert body["points"][-1]["is_anomaly"] is True
    assert body["points"][-1]["index"] == len(block) - 1


@pytest.mark.parametrize(
    "readings",
    [np.zeros((5, 4)).tolist(), np.zeros((30, 3)).tolist()],
    ids=["too-short", "wrong-features"],
)
def test_score_validation(client, readings):
    assert client.post("/score", json={"readings": readings}).status_code == 422
