import numpy as np
import pytest

from iot_anomaly.config import Config
from iot_anomaly.data import make_synthetic_series


@pytest.fixture(scope="session")
def series() -> tuple[np.ndarray, np.ndarray]:
    return make_synthetic_series(n=1200, n_features=4, anomaly_ratio=0.05, seed=1)


@pytest.fixture
def fast_config() -> Config:
    cfg = Config()
    cfg.data.window = 20
    cfg.training.epochs = 15
    cfg.training.batch_size = 64
    cfg.training.device = "cpu"
    cfg.model.hidden_dim = 16
    cfg.model.latent_dim = 4
    return cfg
