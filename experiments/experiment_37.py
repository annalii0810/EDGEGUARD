
"""
EDGEGUARD - Experiment 37
Synthetic Risk-Control Effectiveness

Exploratory research only.
No historical test-period data is used.
"""

import numpy as np
import pandas as pd

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.backtest import run_backtest

SEED = 2037
SIMULATIONS = 500
OBSERVATIONS = 500
CHANGE_POINT = 250

REFERENCE_MEAN = 0.0002299069244105812
DETERIORATED_MEAN = -0.003
VOLATILITY = 0.00715

WINDOW = 50
THRESHOLD = 24.408925
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0


def make_returns(rng, deteriorated):
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


def metrics_for(returns, exposure):
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
    n = len(returns)
    baseline_exposure = np.ones(n)

    _, alarm = historical_adaptive_cusum(
        returns,
        baseline_exposure,
        window=WINDOW,
        target_mean=REFERENCE_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    edge_exposure = baseline_exposure.copy()

    if alarm is not None:
        edge_exposure[alarm + 1:] = REDUCED_EXPOSURE

    # Oracle benchmark: knows the true change point.
    oracle_exposure = baseline_exposure.copy()
    oracle_exposure[CHANGE_POINT:] = REDUCED_EXPOSURE

    return {
        "baseline": metrics_for(
            returns, baseline_exposure
        ),
        "edgeguard": metrics_for(
            returns, edge_exposure
        ),
        "oracle": metrics_for(
            returns, oracle_exposure
        ),
        "alarm": alarm,
    }


def main():
    rng = np.random.default_rng(SEED)

    print("EDGEGUARD - Experiment 37")
    print("Synthetic Risk-Control Effectiveness")
    print("=" * 70)

    for scenario in ["No change", "Deterioration"]:
        deteriorated = scenario == "Deterioration"

        records = {
            name: {
                "return": [],
                "drawdown": [],
                "sharpe": [],
            }
            for name in ["baseline", "edgeguard", "oracle"]
        }

        alarms = 0

        for _ in range(SIMULATIONS):
            returns = make_returns(rng, deteriorated)
            result = evaluate_path(returns)

            if result["alarm"] is not None:
                alarms += 1

            for name in records:
                metrics = result[name]

                records[name]["return"].append(
                    metrics["total_return"]
                )
                records[name]["drawdown"].append(
                    metrics["max_drawdown"]
                )
                records[name]["sharpe"].append(
                    metrics["annualized_sharpe"]
                )

        print(f"\n{scenario}")
        print("-" * 70)
        print("Paths:", SIMULATIONS)
        print("Paths with alarms:", alarms)

        for name, values in records.items():
            print(f"\n{name.upper()}")
            print(
                "Mean total return:",
                f"{np.mean(values['return']):+.2%}",
            )
            print(
                "Mean maximum drawdown:",
                f"{np.mean(values['drawdown']):.2%}",
            )
            print(
                "Mean Sharpe:",
                f"{np.nanmean(values['sharpe']):.3f}",
            )

        baseline = records["baseline"]
        edge = records["edgeguard"]
        oracle = records["oracle"]

        return_difference = (
            np.asarray(edge["return"])
            - np.asarray(baseline["return"])
        )

        drawdown_reduction = (
            np.asarray(baseline["drawdown"])
            - np.asarray(edge["drawdown"])
        )

        oracle_return_difference = (
            np.asarray(oracle["return"])
            - np.asarray(baseline["return"])
        )

        print("\nEDGEGUARD vs baseline")
        print(
            "Mean return difference:",
            f"{np.mean(return_difference):+.2%}",
        )
        print(
            "Mean drawdown reduction:",
            f"{np.mean(drawdown_reduction):+.2%}",
        )
        print(
            "Paths with lower drawdown:",
            f"{np.mean(drawdown_reduction > 0):.2%}",
        )

        print("\nOracle vs baseline")
        print(
            "Mean return difference:",
            f"{np.mean(oracle_return_difference):+.2%}",
        )

    print("\nExploratory simulation only.")
    print("Original Phase 4 results remain unchanged.")


if __name__ == "__main__":
    main()
