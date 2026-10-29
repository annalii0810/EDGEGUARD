
import numpy as np
import pytest

from edgeguard.baseline_detectors import (
    rolling_mean_detector,
    standardized_cusum_detector,
)
from edgeguard.statistical_evaluation import wilson_interval


def test_wilson_interval_bounds():
    low, high = wilson_interval(16, 500)

    assert 0 <= low <= 16 / 500 <= high <= 1
    assert np.isclose(low, 0.019791513626571118)


def test_wilson_interval_extremes():
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0.0)
    assert high > 0

    low, high = wilson_interval(100, 100)
    assert low < 1
    assert high == pytest.approx(1.0)


def test_wilson_invalid_counts():
    for successes, trials in [
        (-1, 100),
        (101, 100),
        (1, 0),
    ]:
        with pytest.raises(ValueError):
            wilson_interval(successes, trials)


def test_rolling_mean_detects_deterioration():
    returns = np.concatenate([
        np.zeros(60),
        np.full(60, -0.01),
    ])

    statistics, alarm = rolling_mean_detector(
        returns,
        window=20,
        target_mean=0.0,
        threshold=2.0,
        reference_volatility=0.00715,
    )

    assert alarm is not None
    assert alarm >= 60
    assert len(statistics) == len(returns)


def test_standard_cusum_detects_deterioration():
    returns = np.concatenate([
        np.zeros(60),
        np.full(60, -0.01),
    ])

    statistics, alarm = standardized_cusum_detector(
        returns,
        target_mean=0.0,
        reference_volatility=0.00715,
        allowance=0.1,
        threshold=10.0,
        warmup=20,
    )

    assert alarm is not None
    assert alarm >= 60
    assert len(statistics) == len(returns)


def test_detectors_reject_invalid_data():
    invalid = np.array([0.01, np.nan, -0.02])

    with pytest.raises(ValueError):
        rolling_mean_detector(invalid, window=2)

    with pytest.raises(ValueError):
        standardized_cusum_detector(invalid, warmup=1)


def test_no_future_information():
    returns = np.zeros(100)
    changed = returns.copy()
    changed[-1] = -0.5

    for detector in [
        lambda x: rolling_mean_detector(
            x, window=20
        ),
        lambda x: standardized_cusum_detector(
            x, warmup=20
        ),
    ]:
        original, _ = detector(returns)
        modified, _ = detector(changed)

        assert np.allclose(
            original[:-1],
            modified[:-1],
        )
