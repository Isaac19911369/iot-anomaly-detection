import numpy as np
import pytest

from iot_anomaly.config import Config
from iot_anomaly.metrics import roc_auc
from iot_anomaly.models import MODEL_NAMES
from iot_anomaly.pipeline import AnomalyPipeline

FEATURES = ["f0", "f1", "f2", "f3"]


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_detectors_separate_injected_anomalies(model, series, fast_config):
    x, y = series
    fast_config.model.name = model
    split = 600  # synthetic anomalies are only injected in the second half
    assert y[:split].sum() == 0
    pipe = AnomalyPipeline(fast_config, FEATURES).fit(x[:split])
    scores = pipe.score_points(x)[split:]
    assert roc_auc(y[split:], scores) > 0.75, model
    assert pipe.threshold is not None and np.isfinite(pipe.threshold)


@pytest.mark.parametrize("model", ["pca", "conv_ae"])
def test_save_load_roundtrip(model, series, fast_config, tmp_path):
    x, _ = series
    fast_config.model.name = model
    pipe = AnomalyPipeline(fast_config, FEATURES).fit(x[:600])
    pipe.save(tmp_path)
    loaded = AnomalyPipeline.load(tmp_path)
    np.testing.assert_allclose(
        loaded.score_windows(x[600:700]), pipe.score_windows(x[600:700]), rtol=1e-5
    )
    assert loaded.threshold == pytest.approx(pipe.threshold)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"nmae": "x"}})


def test_config_loads_default_yaml():
    cfg = Config.load("configs/default.yaml")
    assert cfg.model.name in MODEL_NAMES
    assert cfg.data.window == 60
