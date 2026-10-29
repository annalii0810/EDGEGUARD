
"""
EDGEGUARD - Experiment 40
Volatility Estimation Error Sensitivity

Exploratory synthetic comparison.
No historical test data is used.
"""

import numpy as np
import pandas as pd

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.baseline_detectors import standardized_cusum_detector
from edgeguard.backtest import run_backtest

SEED = 2040
SIMULATIONS = 500
OBSERVATIONS = 500
CHANGE_POINT = 250

REFERENCE_MEAN = 0.0002299069244105812
DETERIORATED_MEAN = -0.003
TRUE_VOLATILITY = 0.00715

WINDOW = 50
ALLOWANCE = 0.1

EDGE_THRESHOLD = 24.408925
STANDARD_THRESHOLD = 23.387877

VOLATILITY_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0


def simulate_path(rng):
    returns = rng.normal(
        REFERENCE_MEAN,
        TRUE_VOLATILITY,
        OBSERVATIONS,
    )

    returns[CHANGE_POINT:] = rng.normal(
        DETERIORATED_MEAN,
        TRUE_VOLATILITY,
        OBSERVATIONS - CHANGE_POINT,
    )

    return returns


def evaluate_returns(returns, alarm):
    exposure = np.ones(len(returns))

    if alarm is not None:
        exposure[alarm + 1:] = REDUCED_EXPOSURE

    data = pd.DataFrame({
        "Return": returns,
        "Exposure": exposure,
    })

    _, metrics = run_backtest(
        data,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=0.0,
    )

    return metrics


def edgeguard_alarm(returns):
    return historical_adaptive_cusum(
        returns,
        np.ones(len(returns)),
        window=WINDOW,
        target_mean=REFERENCE_MEAN,
        allowance=ALLOWANCE,
        threshold=EDGE_THRESHOLD,
        min_volatility=0.001,
    )[1]


def standard_alarm(returns, assumed_volatility):
    return standardized_cusum_detector(
        returns,
        target_mean=REFERENCE_MEAN,
        reference_volatility=assumed_volatility,
        allowance=ALLOWANCE,
        threshold=STANDARD_THRESHOLD,
        warmup=WINDOW,
    )[1]


def main():
    rng = np.random.default_rng(SEED)

    paths = [
        simulate_path(rng)
        for _ in range(SIMULATIONS)
    ]

    edge_metrics = [
        evaluate_returns(path, edgeguard_alarm(path))
        for path in paths
    ]

    print("EDGEGUARD - Experiment 40")
    print("Volatility Estimation Error Sensitivity")
    print("=" * 80)

    print("Simulations:", SIMULATIONS)
    print("True volatility:", TRUE_VOLATILITY)
    print("Frozen EDGEGUARD threshold:", EDGE_THRESHOLD)

    print("\nResults")
    print("=" * 80)

    print(
        f"{'Vol Mult':>10} "
        f"{'Standard Return':>17} "
        f"{'EDGE Return':>14} "
        f"{'EDGE Advantage':>16} "
        f"{'Std Alarm Rate':>15}"
    )

    for multiplier in VOLATILITY_MULTIPLIERS:
        assumed_volatility = (
            TRUE_VOLATILITY * multiplier
        )

        standard_metrics = []
        alarms = 0

        for path in paths:
            alarm = standard_alarm(
                path, assumed_volatility
            )

            if alarm is not None:
                alarms += 1

            standard_metrics.append(
                evaluate_returns(path, alarm)
            )

        standard_returns = np.array([
            m["total_return"]
            for m in standard_metrics
        ])

        edge_returns = np.array([
            m["total_return"]
            for m in edge_metrics
        ])

        advantage = np.mean(
            edge_returns - standard_returns
        )

        print(
            f"{multiplier:>10.2f} "
            f"{np.mean(standard_returns):>+17.2%} "
            f"{np.mean(edge_returns):>+14.2%} "
            f"{advantage:>+16.2%} "
            f"{alarms / SIMULATIONS:>15.1%}"
        )

    print("\nExploratory sensitivity analysis complete.")
    print("Original historical results remain unchanged.")


if __name__ == "__main__":
    main()
