
import numpy as np
import pytest

from edgeguard.detectors import rolling_mean_detector, cusum_detector


def test_rolling_mean_values():
    returns = [1, 2, 3, 4, 5]

    means, alarm = rolling_mean_detector(
        returns, window=3, threshold=0
    )

    assert np.isnan(means[0])
    assert np.isnan(means[1])
    assert np.allclose(means[2:], [2, 3, 4])
    assert alarm is None



def test_first_alarm():
    returns = [0.02, 0.01, -0.04, -0.05]

    means, alarm = rolling_mean_detector(
        returns, window=2
    )

    assert alarm == 2
    assert np.isclose(means[2], -0.015)



def test_no_alarm():
    returns = [0.01, 0.02, 0.03]

    means, alarm = rolling_mean_detector(
        returns, window=2
    )

    assert alarm is None
    assert np.all(means[1:] > 0)



def test_no_future_information():
    original = [0.02, 0.02, 0.02, -0.01]
    modified = [0.02, 0.02, 0.02, -0.50]

    a, _ = rolling_mean_detector(original, window=2)
    b, _ = rolling_mean_detector(modified, window=2)

    assert np.allclose(a[:3], b[:3], equal_nan=True)



def test_invalid_window():
    with pytest.raises(ValueError):
        rolling_mean_detector([0.01, 0.02], window=3)


def test_invalid_data():
    with pytest.raises(ValueError):
        rolling_mean_detector([0.01, np.nan], window=1)



def test_cusum_detects_deterioration():
    returns = [0.01, 0.01, -0.02, -0.02]

    values, alarm = cusum_detector(
        returns,
        target_mean=0.01,
        allowance=0.005,
        threshold=0.03
    )

    assert alarm == 3
    assert np.isclose(values[2], 0.025)
    assert np.isclose(values[3], 0.050)



def test_cusum_no_alarm():
    returns = [0.02, 0.03, 0.02, 0.04]

    values, alarm = cusum_detector(
        returns,
        target_mean=0.01,
        allowance=0.005,
        threshold=0.03
    )

    assert alarm is None
    assert np.allclose(values, 0.0)



def test_cusum_invalid_inputs():
    with pytest.raises(ValueError):
        cusum_detector([], threshold=0.03)

    with pytest.raises(ValueError):
        cusum_detector([0.01, np.nan], threshold=0.03)

    with pytest.raises(ValueError):
        cusum_detector([0.01], allowance=-0.005)

    with pytest.raises(ValueError):
        cusum_detector([0.01], threshold=-0.03)

    with pytest.raises(ValueError):
        cusum_detector([0.01], threshold=0)



def test_cusum_no_future_information():
    original = [0.01, -0.01, -0.02, -0.01]
    modified = [0.01, -0.01, -0.02, -0.50]

    a, _ = cusum_detector(
        original,
        target_mean=0.01,
        allowance=0.005,
        threshold=0.03
    )

    b, _ = cusum_detector(
        modified,
        target_mean=0.01,
        allowance=0.005,
        threshold=0.03
    )

    assert np.allclose(a[:3], b[:3])
