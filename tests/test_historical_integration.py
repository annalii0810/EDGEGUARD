
import numpy as np
import pandas as pd

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.risk_controller import apply_risk_control
from edgeguard.backtest import run_backtest


def evaluate(returns, exposure):
    strategy_returns = returns * exposure

    stats, alarm = historical_adaptive_cusum(
        strategy_returns,
        exposure,
        window=20,
        target_mean=0.001,
        allowance=0.1,
        threshold=5.0,
        min_volatility=0.001,
    )

    _, multiplier = apply_risk_control(
        strategy_returns,
        alarm_index=alarm,
        reduced_exposure=0.25,
    )

    data = pd.DataFrame({
        "Return": returns,
        "Exposure": exposure * multiplier,
    })

    result, _ = run_backtest(
        data,
        transaction_cost_bps=5.0,
        initial_exposure=0.0,
    )

    return result, stats, alarm


def test_future_returns_do_not_change_past_results():
    rng = np.random.default_rng(2026)

    returns = rng.normal(0.001, 0.01, 150)
    exposure = np.ones(150)

    original, original_stats, _ = evaluate(
        returns, exposure
    )

    modified_returns = returns.copy()
    modified_returns[100:] = -0.05

    modified, modified_stats, _ = evaluate(
        modified_returns, exposure
    )

    assert np.allclose(
        original_stats[:100],
        modified_stats[:100],
    )

    assert np.allclose(
        original["NetReturn"].iloc[:100],
        modified["NetReturn"].iloc[:100],
    )


def test_alarm_reduces_exposure_next_observation():
    returns = np.concatenate([
        np.full(40, 0.001),
        np.full(40, -0.02),
    ])

    exposure = np.ones(len(returns))

    result, _, alarm = evaluate(
        returns, exposure
    )

    assert alarm is not None
    assert result["Exposure"].iloc[alarm] == 1.0

    if alarm + 1 < len(result):
        assert result["Exposure"].iloc[alarm + 1] == 0.25


def test_no_alarm_preserves_baseline():
    returns = np.full(100, 0.001)
    exposure = np.ones(100)

    result, _, alarm = evaluate(
        returns, exposure
    )

    assert alarm is None

    assert np.allclose(
        result["Exposure"], exposure
    )
