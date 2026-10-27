from edgeguard.detectors import adaptive_cusum_detector
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



def test_adaptive_cusum_detects_deterioration():
    # First three returns establish historical volatility.
    # Later negative returns represent deterioration.
    returns = [
        0.01, 0.02, 0.03,
        -0.02, -0.02, -0.02
    ]

    values, alarm = adaptive_cusum_detector(
        returns,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=2.0,
    )

    assert len(values) == len(returns)
    assert alarm == 3
    assert np.all(values[:3] == 0.0)
    assert np.isclose(values[3], 2.9)



def test_adaptive_cusum_no_future_information():
    original = [
        0.01, 0.02, 0.03, 0.04,
        -0.02, -0.01, 0.01, 0.02
    ]

    modified = [
        0.01, 0.02, 0.03, 0.04,
        -0.02, 0.50, -0.40, 0.30
    ]

    a, alarm_a = adaptive_cusum_detector(
        original,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=2.0
    )

    b, alarm_b = adaptive_cusum_detector(
        modified,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=2.0
    )

    # Future changes cannot affect earlier statistics.
    assert np.allclose(a[:5], b[:5])

    # Both sequences trigger the same first alarm.
    assert alarm_a == alarm_b == 4



def test_adaptive_cusum_invalid_inputs():
    returns = [0.01, 0.02, 0.03, 0.04]

    # Volatility window must be valid
    with pytest.raises(ValueError):
        adaptive_cusum_detector(returns, window=1)

    with pytest.raises(ValueError):
        adaptive_cusum_detector(returns, window=4)

    # Reject invalid return observations
    with pytest.raises(ValueError):
        adaptive_cusum_detector(
            [0.01, np.nan, 0.03, 0.04],
            window=2
        )

    # Reject negative allowance
    with pytest.raises(ValueError):
        adaptive_cusum_detector(
            returns,
            window=2,
            allowance=-0.1
        )

    # Reject invalid thresholds
    with pytest.raises(ValueError):
        adaptive_cusum_detector(
            returns,
            window=2,
            threshold=0
        )

    # Reject invalid volatility floor
    with pytest.raises(ValueError):
        adaptive_cusum_detector(
            returns,
            window=2,
            min_volatility=0
        )



def test_adaptive_cusum_zero_volatility():
    returns = [
        0.01, 0.01, 0.01,
        0.01, 0.01, 0.01
    ]

    values, alarm = adaptive_cusum_detector(
        returns,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=2.0,
        min_volatility=1e-6
    )

    # The detector must not produce NaN or infinity.
    assert np.all(np.isfinite(values))

    # Returns equal the target mean, so no alarm is expected.
    assert alarm is None

    # CUSUM should remain at zero.
    assert np.allclose(values, 0.0)



def test_adaptive_cusum_near_zero_volatility():
    returns = [
        0.01, 0.01, 0.01,
        0.0099, 0.01, 0.01
    ]

    values, alarm = adaptive_cusum_detector(
        returns,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=25.0,
        min_volatility=1e-6
    )

    # The calculations must remain finite.
    assert np.all(np.isfinite(values))

    # A small deviation becomes very large
    # when divided by the volatility floor.
    assert np.isclose(values[3], 99.9)

    # Record the behavior of the current implementation.
    assert alarm == 3


def test_adaptive_cusum_volatility_floor_effect():
    returns = [
        0.01, 0.01, 0.01,
        0.0099, 0.01, 0.01
    ]

    small_floor, small_alarm = adaptive_cusum_detector(
        returns,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=25.0,
        min_volatility=1e-6,
    )

    larger_floor, larger_alarm = adaptive_cusum_detector(
        returns,
        window=3,
        target_mean=0.01,
        allowance=0.1,
        threshold=25.0,
        min_volatility=0.001,
    )

    assert np.isclose(small_floor[3], 99.9)
    assert np.isclose(larger_floor[3], 0.0)

    assert small_alarm == 3
    assert larger_alarm is None

    assert np.all(np.isfinite(small_floor))
    assert np.all(np.isfinite(larger_floor))
