
"""
EDGEGUARD - Experiment 30
Final Out-of-Sample Historical Evaluation

Frozen methodology commit: 12b293d

IMPORTANT:
Do not run on reserved test data until
pre-execution review is complete.
"""

import hashlib
import json
from pathlib import Path

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

# Frozen detector parameters
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001

REDUCED_EXPOSURE = 0.25
TRANSACTION_COST_BPS = 5.0

FROZEN_COMMIT = "12b293d"

REPRODUCIBILITY_FILE = Path(
    "docs/reproducibility.md"
)


def verify_dataset_fingerprint(ticker):
    """
    Check the dataset against the frozen
    reproducibility record.
    """

    path = Path(f"data/raw/{ticker}.csv")

    if not path.exists():
        raise FileNotFoundError(path)

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    record = REPRODUCIBILITY_FILE.read_text()

    if digest not in record:
        raise ValueError(
            f"{ticker} dataset fingerprint mismatch."
        )

    return digest


def evaluate_asset(ticker):
    verify_dataset_fingerprint(ticker)

    prices = load_historical_data(
        f"data/raw/{ticker}.csv"
    )

    strategy = moving_average_strategy(
        prices,
        window=STRATEGY_WINDOW,
    )

    train, validation, test = chronological_split(
        strategy
    )

    # Test-period trading position inherits
    # the final validation strategy position.
    initial_exposure = float(
        validation["Exposure"].iloc[-1]
    )

    strategy_returns = test[
        "StrategyReturn"
    ].to_numpy(dtype=float)

    strategy_exposure = test[
        "Exposure"
    ].to_numpy(dtype=float)

    statistics, alarm = historical_adaptive_cusum(
        strategy_returns,
        strategy_exposure,
        window=DETECTOR_WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    # Detector resets at the test boundary.
    # An alarm changes exposure beginning at t+1.
    _, multiplier = apply_risk_control(
        strategy_returns,
        alarm_index=alarm,
        reduced_exposure=REDUCED_EXPOSURE,
    )

    controlled = test.copy()

    controlled["Exposure"] = (
        strategy_exposure * multiplier
    )

    _, baseline = run_backtest(
        test,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=initial_exposure,
    )

    _, protected = run_backtest(
        controlled,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_exposure=initial_exposure,
    )

    return {
        "ticker": ticker,
        "observations": len(test),
        "first_date": str(
            test["Date"].iloc[0].date()
        ),
        "last_date": str(
            test["Date"].iloc[-1].date()
        ),
        "initial_exposure": initial_exposure,
        "first_test_exposure": float(
            strategy_exposure[0]
        ),
        "alarm_index": (
            int(alarm) if alarm is not None else None
        ),
        "alarm_date": (
            str(test["Date"].iloc[alarm].date())
            if alarm is not None else None
        ),
        "maximum_cusum": float(
            np.max(statistics)
        ),
        "baseline": baseline,
        "controlled": protected,
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
    print("EDGEGUARD - Experiment 30")
    print("Final Out-of-Sample Evaluation")
    print("=" * 75)

    print("Frozen commit:", FROZEN_COMMIT)
    print("Detector threshold:", THRESHOLD)
    print("Target mean:", TARGET_MEAN)
    print("Transaction costs:", TRANSACTION_COST_BPS)
    print("Detector reset: each evaluation period")

    results = []

    for ticker in TICKERS:
        print(f"\nEvaluating {ticker}...")
        result = evaluate_asset(ticker)
        results.append(result)

    print("\nFinal Test Results")
    print("=" * 75)

    for result in results:
        print(f"\n{result['ticker']}")
        print(
            "Test dates:",
            result["first_date"],
            "to",
            result["last_date"],
        )
        print(
            "Initial exposure:",
            result["initial_exposure"],
        )
        print(
            "First test exposure:",
            result["first_test_exposure"],
        )
        print(
            "Alarm date:",
            result["alarm_date"],
        )
        print(
            "Maximum CUSUM:",
            f"{result['maximum_cusum']:.3f}",
        )
        print(
            "Baseline return:",
            f"{result['baseline']['total_return']:.2%}",
        )
        print(
            "EDGEGUARD return:",
            f"{result['controlled']['total_return']:.2%}",
        )
        print(
            "Return difference:",
            f"{result['return_difference']:+.2%}",
        )
        print(
            "Drawdown reduction:",
            f"{result['drawdown_reduction']:+.2%}",
        )
        print(
            "Sharpe difference:",
            f"{result['sharpe_difference']:+.3f}",
        )

    print("\n" + "=" * 75)
    print("Final evaluation completed.")
    print("Frozen parameters were not modified.")


if __name__ == "__main__":
    main()
