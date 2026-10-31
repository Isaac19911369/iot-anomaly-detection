import numpy as np
import pytest

from iot_anomaly.metrics import detection_metrics, roc_auc


def test_detection_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    m = detection_metrics(y, p)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    assert m["far"] == pytest.approx(25.0)
    assert m["mar"] == pytest.approx(25.0)


def test_no_predictions_is_safe():
    m = detection_metrics(np.array([0, 1]), np.array([0, 0]))
    assert m["f1"] == 0.0 and m["mar"] == 100.0


def test_roc_auc_single_class_is_nan():
    assert np.isnan(roc_auc(np.zeros(5), np.arange(5)))
