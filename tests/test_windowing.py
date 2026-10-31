import numpy as np
import pytest

from iot_anomaly.data import StandardScaler, make_windows, window_scores_to_points


def test_make_windows_shape_and_content():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    w = make_windows(x, window=4)
    assert w.shape == (7, 4, 2)
    np.testing.assert_array_equal(w[0], x[:4])
    np.testing.assert_array_equal(w[-1], x[-4:])


def test_make_windows_stride():
    x = np.zeros((10, 3))
    assert make_windows(x, window=4, stride=3).shape == (3, 4, 3)


def test_make_windows_too_short():
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 2)), window=4)


def test_points_alignment():
    pts = window_scores_to_points(np.array([1.0, 2.0, 3.0]), window=3)
    assert len(pts) == 5
    assert np.isnan(pts[:2]).all()
    np.testing.assert_array_equal(pts[2:], [1.0, 2.0, 3.0])


def test_scaler_roundtrip_and_constant_feature():
    x = np.c_[np.random.default_rng(0).normal(5, 2, 100), np.full(100, 3.0)]
    sc = StandardScaler().fit(x)
    z = sc.transform(x)
    assert abs(z[:, 0].mean()) < 1e-6 and abs(z[:, 0].std() - 1) < 1e-6
    assert np.allclose(z[:, 1], 0)  # constant feature does not divide by zero
    restored = StandardScaler.from_dict(sc.to_dict())
    np.testing.assert_allclose(restored.transform(x), z)
