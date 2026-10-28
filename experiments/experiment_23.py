
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control
from edgeguard.backtest import run_backtest


DATA_PATH = "data/raw/SPY.csv"

STRATEGY_WINDOW = 50
DETECTOR_WINDOW = 50

TARGET_MEAN = 0.001
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 1e-6

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 40)
    print(f"Total return: {metrics['total_return']:.2%}")
    print(f"Maximum drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe ratio: {metrics['annualized_sharpe']:.3f}")
    print(f"Total turnover: {metrics['total_turnover']:.2f}")


def evaluate_period(name, data):
    print(f"\n{'=' * 55}")
    print(name)
    print(f"Observations: {len(data)}")

    strategy_returns = data["StrategyReturn"].to_numpy()

    active = np.flatnonzero(
        data["Exposure"].to_numpy() > 0
    )

    print(
    "First active strategy observation:",
    int(active[0]) if len(active) else "None"
    )

    statistics, alarm_index = adaptive_cusum_detector(
        strategy_returns,
        window=DETECTOR_WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    _, control_multiplier = apply_risk_control(
        strategy_returns,
        alarm_index=alarm_index,
        reduced_exposure=REDUCED_EXPOSURE,
    )

    # Preserve the moving-average strategy's positions.
    controlled = data.copy()
    controlled["Exposure"] = (
        data["Exposure"].to_numpy()
        * control_multiplier
    )

    # Match exposure at the beginning of the period
    # to avoid charging an artificial entry cost.
    initial_exposure = float(data["Exposure"].iloc[0])

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

    if alarm_index is None:
        print("\nFirst alarm: None")
    else:
        alarm_date = data["Date"].iloc[alarm_index]
        print(f"\nFirst alarm: {alarm_date.date()}")
        print(f"Alarm index: {alarm_index}")
        print(f"CUSUM statistic: {statistics[alarm_index]:.4f}")

    print_metrics("Original Moving-Average Strategy", baseline_metrics)
    print_metrics("EDGEGUARD-Controlled Strategy", controlled_metrics)

    print("\nComparison")
    print("-" * 40)

    return_difference = (
        controlled_metrics["total_return"]
        - baseline_metrics["total_return"]
    )

    drawdown_reduction = (
        baseline_metrics["max_drawdown"]
        - controlled_metrics["max_drawdown"]
    )

    print(f"Return difference: {return_difference:+.2%}")
    print(f"Drawdown reduction: {drawdown_reduction:+.2%}")


def main():
    print("EDGEGUARD - Experiment 23")
    print("Historical Adaptive CUSUM Integration")
    print("=" * 55)

    prices = load_historical_data(DATA_PATH)

    strategy = moving_average_strategy(
        prices,
        window=STRATEGY_WINDOW,
    )

    train, validation, _ = chronological_split(strategy)

    evaluate_period("Training (2010-2018)", train)
    evaluate_period("Validation (2019-2022)", validation)


if __name__ == "__main__":
    main()
