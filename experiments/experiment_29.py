
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.risk_controller import apply_risk_control
from edgeguard.backtest import run_backtest


TICKERS = ["SPY", "QQQ", "IWM", "TLT"]

STRATEGY_WINDOW = 50
DETECTOR_WINDOW = 50

# Frozen SPY training-calibrated parameters
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0


def evaluate_period(ticker, period, data, initial_exposure):
    returns = data["StrategyReturn"].to_numpy(dtype=float)
    exposure = data["Exposure"].to_numpy(dtype=float)

    statistics, alarm = historical_adaptive_cusum(
        returns,
        exposure,
        window=DETECTOR_WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    _, multiplier = apply_risk_control(
        returns,
        alarm_index=alarm,
        reduced_exposure=REDUCED_EXPOSURE,
    )

    controlled = data.copy()
    controlled["Exposure"] = exposure * multiplier


    _, baseline = run_backtest(
        data,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=initial_exposure,
    )

    _, protected = run_backtest(
        controlled,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=initial_exposure,
    )

    peak = int(np.argmax(statistics))

    return {
        "ticker": ticker,
        "period": period,
        "observations": len(data),
        "active": int(np.sum(exposure > 0)),
        "alarm": alarm,
        "alarm_date": (
            str(data["Date"].iloc[alarm].date())
            if alarm is not None else "None"
        ),
        "maximum_cusum": float(statistics[peak]),
        "baseline_return": baseline["total_return"],
        "controlled_return": protected["total_return"],
        "return_difference": (
            protected["total_return"]
            - baseline["total_return"]
        ),
        "drawdown_reduction": (
            baseline["max_drawdown"]
            - protected["max_drawdown"]
        ),
        "sharpe_difference": (
            protected["annualized_sharpe"]
            - baseline["annualized_sharpe"]
        ),
    }


def main():
    print("EDGEGUARD - Experiment 29")
    print("Multi-Asset Historical Robustness")
    print("=" * 80)

    print("Detector threshold:", THRESHOLD)
    print("Reference mean:", TARGET_MEAN)
    print("Post-alarm exposure:", REDUCED_EXPOSURE)
    print("Transaction costs:", TRANSACTION_COST_BPS, "bps")
    print("Final test period excluded.")

    results = []

    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")

        prices = load_historical_data(
            f"data/raw/{ticker}.csv"
        )

        strategy = moving_average_strategy(
            prices,
            window=STRATEGY_WINDOW,
        )

        train, validation, _ = chronological_split(strategy)

        
        for period, data, initial_exposure in [
            ("Training", train, 0.0),
            (
                "Validation",
                validation,
                float(train["Exposure"].iloc[-1]),
            ),
        ]:
            result = evaluate_period(
                ticker,
                period,
                data,
                initial_exposure,
            )

            results.append(result)


    print("\nDetection Results")
    print("=" * 80)

    print(
        f"{'Asset':<8} {'Period':<12} "
        f"{'Active':>8} {'Max CUSUM':>12} "
        f"{'Alarm Date':>15}"
    )

    for r in results:
        print(
            f"{r['ticker']:<8} "
            f"{r['period']:<12} "
            f"{r['active']:>8} "
            f"{r['maximum_cusum']:>12.3f} "
            f"{r['alarm_date']:>15}"
        )

    print("\nTrading Performance Differences")
    print("=" * 80)

    print(
        f"{'Asset':<8} {'Period':<12} "
        f"{'Return Diff':>13} "
        f"{'DD Reduction':>15} "
        f"{'Sharpe Diff':>13}"
    )

    for r in results:
        print(
            f"{r['ticker']:<8} "
            f"{r['period']:<12} "
            f"{r['return_difference']:>+13.2%} "
            f"{r['drawdown_reduction']:>+15.2%} "
            f"{r['sharpe_difference']:>+13.3f}"
        )

    print("\nSummary")
    print("=" * 80)

    print("Total asset-period evaluations:", len(results))
    print(
        "Periods with alarms:",
        sum(r["alarm"] is not None for r in results)
    )
    print(
        "Periods with positive return differences:",
        sum(r["return_difference"] > 0 for r in results)
    )
    print(
        "Periods with drawdown reductions:",
        sum(r["drawdown_reduction"] > 0 for r in results)
    )


if __name__ == "__main__":
    main()
