
import numpy as np
import pytest

from edgeguard.historical_detector import historical_adaptive_cusum


def detect(returns, exposure, **kwargs):
    return historical_adaptive_cusum(
        returns,
        exposure,
        window=5,
        target_mean=0.001,
        allowance=0.1,
        threshold=5.0,
        min_volatility=0.001,
        **kwargs,
    )



def test_output_length():
    returns = np.zeros(20)
    exposure = np.ones(20)

    statistics, alarm = detect(returns, exposure)

    assert len(statistics) == 20
    assert np.isfinite(statistics).all()
    assert alarm is None or 0 <= alarm < len(returns)



def test_inactive_periods():
    returns = np.zeros(20)
    exposure = np.zeros(20)

    statistics, alarm = detect(returns, exposure)

    assert np.all(statistics == 0)
    assert alarm is None


def test_no_alarm_before_active_warmup():
    returns = np.array(
        [0.0] * 10 + [-0.01] * 5
    )
    exposure = np.array(
        [0.0] * 10 + [1.0] * 5
    )

    statistics, alarm = detect(returns, exposure)

    assert alarm is None
    assert np.all(statistics == 0)


def test_zero_volatility_is_finite():
    returns = np.array(
        [0.01] * 6 + [-0.01]
    )
    exposure = np.ones(len(returns))

    statistics, alarm = detect(returns, exposure)

    assert np.isfinite(statistics).all()
    assert alarm is not None


def test_alarm_after_deterioration():
    returns = np.array(
        [0.01] * 10 + [-0.02] * 10
    )
    exposure = np.ones(len(returns))

    _, alarm = detect(returns, exposure)

    assert alarm is not None
    assert alarm >= 5


def test_no_future_information():
    returns = np.array(
        [0.01] * 10 + [-0.02] * 10
    )
    exposure = np.ones(len(returns))

    original, original_alarm = detect(
        returns, exposure
    )

    modified = returns.copy()
    modified[15:] = 0.05

    changed, changed_alarm = detect(
        modified, exposure
    )

    assert np.allclose(
        original[:15],
        changed[:15]
    )

    if original_alarm is not None and original_alarm < 15:
        assert changed_alarm == original_alarm


def test_exposure_normalization():
    raw_returns = np.array(
        [0.01] * 10 + [-0.02] * 10
    )

    full_exposure = np.ones(len(raw_returns))
    half_exposure = np.full(len(raw_returns), 0.5)

    full_statistics, full_alarm = detect(
        raw_returns, full_exposure
    )

    half_statistics, half_alarm = detect(
        raw_returns * 0.5, half_exposure
    )

    assert np.allclose(
        full_statistics,
        half_statistics
    )
    assert full_alarm == half_alarm


def test_invalid_exposure():
    returns = np.zeros(10)
    exposure = np.ones(10)
    exposure[3] = 1.5

    with pytest.raises(ValueError):
        detect(returns, exposure)


def test_invalid_window():
    returns = np.zeros(10)
    exposure = np.ones(10)

    with pytest.raises(ValueError):
        historical_adaptive_cusum(
            returns, exposure, window=1
        )


def test_invalid_threshold():
    returns = np.zeros(10)
    exposure = np.ones(10)

    with pytest.raises(ValueError):
        historical_adaptive_cusum(
            returns, exposure, threshold=0
        )


def test_alarm_does_not_change_current_exposure():
    import numpy as np

    from edgeguard.historical_detector import historical_adaptive_cusum
    from edgeguard.risk_controller import apply_risk_control

    returns = np.concatenate([
        np.full(60, 0.001),
        np.full(40, -0.02),
    ])

    exposure = np.ones(len(returns))

    _, alarm = historical_adaptive_cusum(
        returns,
        exposure,
        window=20,
        target_mean=0.001,
        allowance=0.1,
        threshold=5.0,
        min_volatility=0.001,
    )

    assert alarm is not None

    _, multiplier = apply_risk_control(
        returns,
        alarm_index=alarm,
        reduced_exposure=0.25,
    )

    assert multiplier[alarm] == 1.0
    assert np.all(multiplier[:alarm + 1] == 1.0)

    if alarm + 1 < len(returns):
        assert np.all(multiplier[alarm + 1:] == 0.25)


def test_detector_has_no_future_information_leakage():
    import numpy as np

    from edgeguard.historical_detector import historical_adaptive_cusum

    rng = np.random.default_rng(2026)

    returns = rng.normal(0.0002, 0.01, 120)
    exposure = np.ones(120)

    original, _ = historical_adaptive_cusum(
        returns,
        exposure,
        window=20,
        target_mean=0.0002,
        threshold=25.0,
    )

    modified_returns = returns.copy()
    modified_returns[100:] = -0.05

    modified, _ = historical_adaptive_cusum(
        modified_returns,
        exposure,
        window=20,
        target_mean=0.0002,
        threshold=25.0,
    )

    assert np.allclose(
        original[:100],
        modified[:100],
    )


def test_inactive_period_preserves_statistic():
    import numpy as np

    from edgeguard.historical_detector import historical_adaptive_cusum

    returns = np.concatenate([
        np.full(30, 0.001),
        np.full(5, -0.01),
        np.zeros(10),
    ])

    exposure = np.concatenate([
        np.ones(35),
        np.zeros(10),
    ])

    statistics, _ = historical_adaptive_cusum(
        returns,
        exposure,
        window=20,
        target_mean=0.001,
        threshold=100.0,
    )

    assert statistics[34] > 0

    assert np.allclose(
        statistics[35:],
        statistics[34],
    )

