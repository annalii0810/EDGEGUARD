
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.risk_controller import apply_risk_control
from edgeguard.backtest import run_backtest


DATA_PATH = "data/raw/SPY.csv"

STRATEGY_WINDOW = 50
DETECTOR_WINDOW = 50

# Frozen training-calibrated detector parameters
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 45)

    print(
        f"Total return: "
        f"{metrics['total_return']:.2%}"
    )

    print(
        f"Maximum drawdown: "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"Sharpe ratio: "
        f"{metrics['annualized_sharpe']:.3f}"
    )

    print(
        f"Total turnover: "
        f"{metrics['total_turnover']:.2f}"
    )


def evaluate_period(name, data, initial_exposure):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    strategy_returns = data[
        "StrategyReturn"
    ].to_numpy(dtype=float)

    strategy_exposure = data[
        "Exposure"
    ].to_numpy(dtype=float)

    # Monitor the original strategy's returns.
    statistics, alarm_index = historical_adaptive_cusum(
        strategy_returns,
        strategy_exposure,
        window=DETECTOR_WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    # An alarm at t changes exposure starting at t+1.
    _, control_multiplier = apply_risk_control(
        strategy_returns,
        alarm_index=alarm_index,
        reduced_exposure=REDUCED_EXPOSURE,
    )

    # Preserve the original moving-average positions.
    controlled = data.copy()

    controlled["Exposure"] = (
        strategy_exposure * control_multiplier
    )

    # The backtester uses raw asset returns and applies
    # the exposure exactly once.
    #
    # Initial exposure is supplied by the caller using
    # the preceding period's final position.

    _, baseline_metrics = run_backtest(
        data,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=initial_exposure,
    )

    _, controlled_metrics = run_backtest(
        controlled,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=initial_exposure,
    )

    print(f"Observations: {len(data)}")

    print(
        "Active observations:",
        int(np.sum(strategy_exposure > 0))
    )

    print(
        "Initial exposure:",
        f"{initial_exposure:.2f}"
    )

    if alarm_index is None:
        print("\nFirst alarm: None")

    else:
        alarm_date = data["Date"].iloc[alarm_index]

        print(
            "\nFirst alarm:",
            alarm_date.date()
        )

        print(
            "Alarm index:",
            alarm_index
        )

        print(
            "CUSUM statistic:",
            f"{statistics[alarm_index]:.4f}"
        )

        if alarm_index + 1 < len(data):
            reduction_date = data[
                "Date"
            ].iloc[alarm_index + 1]

            print(
                "Exposure reduction begins:",
                reduction_date.date()
            )

    print_metrics(
        "Original Moving-Average Strategy",
        baseline_metrics,
    )

    print_metrics(
        "Calibrated EDGEGUARD Strategy",
        controlled_metrics,
    )

    return_difference = (
        controlled_metrics["total_return"]
        - baseline_metrics["total_return"]
    )

    drawdown_reduction = (
        baseline_metrics["max_drawdown"]
        - controlled_metrics["max_drawdown"]
    )

    sharpe_difference = (
        controlled_metrics["annualized_sharpe"]
        - baseline_metrics["annualized_sharpe"]
    )

    print("\nPaired Comparison")
    print("-" * 45)

    print(
        f"Return difference: "
        f"{return_difference:+.2%}"
    )

    print(
        f"Drawdown reduction: "
        f"{drawdown_reduction:+.2%}"
    )

    print(
        f"Sharpe difference: "
        f"{sharpe_difference:+.3f}"
    )

    return {
        "period": name,
        "alarm_index": alarm_index,
        "baseline_metrics": baseline_metrics,
        "controlled_metrics": controlled_metrics,
        "return_difference": return_difference,
        "drawdown_reduction": drawdown_reduction,
        "sharpe_difference": sharpe_difference,
    }


def main():
    print("EDGEGUARD - Experiment 28")
    print("Calibrated Historical Risk Controller")
    print("=" * 60)

    prices = load_historical_data(DATA_PATH)

    # Construct strategy signals using chronological
    # historical prices.
    strategy = moving_average_strategy(
        prices,
        window=STRATEGY_WINDOW,
    )

    # Reserve the final test period.
    train, validation, _ = chronological_split(
        strategy
    )

    # Training starts from a flat position.
    evaluate_period(
        "Training (2010-2018)",
        train,
        initial_exposure=0.0,
    )

    # Validation inherits the preceding strategy position.
    validation_initial_exposure = float(
        train["Exposure"].iloc[-1]
    )

    evaluate_period(
        "Validation (2019-2022)",
        validation,
        initial_exposure=validation_initial_exposure,
    )

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("Detector parameters remained frozen.")
    print("Detector state resets at each period.")
    print("Final test period remains untouched.")


if __name__ == "__main__":
    main()
