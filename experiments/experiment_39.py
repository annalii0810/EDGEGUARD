
"""
EDGEGUARD - Experiment 39
Paired Economic Comparison Against Standard CUSUM

Exploratory synthetic research only.
No historical test-period data is used.
"""

import numpy as np
import pandas as pd

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.baseline_detectors import standardized_cusum_detector
from edgeguard.backtest import run_backtest

SEED = 2039
SIMULATIONS = 500
OBSERVATIONS = 500
CHANGE_POINT = 250

REFERENCE_MEAN = 0.0002299069244105812
DETERIORATED_MEAN = -0.003
VOLATILITY = 0.00715

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001

EDGE_THRESHOLD = 24.408925
STANDARD_THRESHOLD = 23.387877

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0
BOOTSTRAPS = 2000


def generate_path(rng, deteriorated):
    returns = rng.normal(
        REFERENCE_MEAN, VOLATILITY, OBSERVATIONS
    )

    if deteriorated:
        returns[CHANGE_POINT:] = rng.normal(
            DETERIORATED_MEAN,
            VOLATILITY,
            OBSERVATIONS - CHANGE_POINT,
        )

    return returns


def backtest(returns, alarm):
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


def evaluate_path(returns):
    edge_alarm = historical_adaptive_cusum(
        returns,
        np.ones(len(returns)),
        window=WINDOW,
        target_mean=REFERENCE_MEAN,
        allowance=ALLOWANCE,
        threshold=EDGE_THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )[1]

    standard_alarm = standardized_cusum_detector(
        returns,
        target_mean=REFERENCE_MEAN,
        reference_volatility=VOLATILITY,
        allowance=ALLOWANCE,
        threshold=STANDARD_THRESHOLD,
        warmup=WINDOW,
    )[1]

    return {
        "EDGEGUARD": backtest(returns, edge_alarm),
        "Standard CUSUM": backtest(returns, standard_alarm),
        "Baseline": backtest(returns, None),
        "edge_alarm": edge_alarm,
        "standard_alarm": standard_alarm,
    }


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)

    estimates = np.empty(BOOTSTRAPS)

    for i in range(BOOTSTRAPS):
        indices = rng.integers(0, n, size=n)
        estimates[i] = np.mean(values[indices])

    return np.quantile(estimates, [0.025, 0.975])


def main():
    rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(2040)

    print("EDGEGUARD - Experiment 39")
    print("Risk-Control Comparison: EDGEGUARD vs Standard CUSUM")
    print("=" * 75)

    for scenario in ["No change", "Deterioration"]:
        deteriorated = scenario == "Deterioration"

        edge_returns = []
        standard_returns = []
        baseline_returns = []

        edge_drawdowns = []
        standard_drawdowns = []

        edge_alarms = 0
        standard_alarms = 0

        for _ in range(SIMULATIONS):
            returns = generate_path(rng, deteriorated)
            result = evaluate_path(returns)

            edge_alarms += result["edge_alarm"] is not None
            standard_alarms += result["standard_alarm"] is not None

            edge_returns.append(
                result["EDGEGUARD"]["total_return"]
            )
            standard_returns.append(
                result["Standard CUSUM"]["total_return"]
            )
            baseline_returns.append(
                result["Baseline"]["total_return"]
            )

            edge_drawdowns.append(
                result["EDGEGUARD"]["max_drawdown"]
            )
            standard_drawdowns.append(
                result["Standard CUSUM"]["max_drawdown"]
            )

        return_diff = (
            np.array(edge_returns)
            - np.array(standard_returns)
        )

        drawdown_diff = (
            np.array(standard_drawdowns)
            - np.array(edge_drawdowns)
        )

        print(f"\n{scenario}")
        print("-" * 75)

        print("Paths:", SIMULATIONS)
        print("EDGEGUARD alarms:", edge_alarms)
        print("Standard CUSUM alarms:", standard_alarms)

        print("\nMean total returns")
        print(f"Baseline: {np.mean(baseline_returns):+.2%}")
        print(f"EDGEGUARD: {np.mean(edge_returns):+.2%}")
        print(f"Standard CUSUM: {np.mean(standard_returns):+.2%}")

        print("\nMean maximum drawdowns")
        print(f"EDGEGUARD: {np.mean(edge_drawdowns):.2%}")
        print(f"Standard CUSUM: {np.mean(standard_drawdowns):.2%}")

        for label, values in [
            ("EDGEGUARD minus Standard CUSUM return", return_diff),
            ("Standard minus EDGEGUARD drawdown", drawdown_diff),
        ]:
            low, high = bootstrap_ci(values, bootstrap_rng)

            print(f"\n{label}")
            print(f"Mean: {np.mean(values):+.2%}")
            print(f"95% paired bootstrap CI: [{low:+.2%}, {high:+.2%}]")
            print(f"Positive-effect paths: {np.mean(values > 0):.2%}")

    print("\nExploratory comparison complete.")
    print("Historical Phase 4 results remain unchanged.")


if __name__ == "__main__":
    main()
