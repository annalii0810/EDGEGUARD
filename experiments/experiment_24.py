
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.historical_detector import historical_adaptive_cusum


DATA_PATH = "data/raw/SPY.csv"

WINDOW = 50
TARGET_MEAN = 0.001
ALLOWANCE = 0.1
THRESHOLD = 25.0


def evaluate(name, data):
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    returns = data["StrategyReturn"].to_numpy()
    exposure = data["Exposure"].to_numpy()

    old_stats, old_alarm = adaptive_cusum_detector(
        returns,
        window=WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=1e-6,
    )

    new_stats, new_alarm = historical_adaptive_cusum(
        returns,
        exposure,
        window=WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=0.001,
    )

    print(f"Observations: {len(returns)}")
    print(f"Active observations: {np.sum(exposure > 0)}")

    print("\nOriginal Adaptive CUSUM")
    print("-" * 35)
    print(f"First alarm index: {old_alarm}")

    if old_alarm is not None:
        print(
            "First alarm date:",
            data["Date"].iloc[old_alarm].date()
        )
        print(
            f"CUSUM statistic: {old_stats[old_alarm]:.4f}"
        )

    print("\nHistorical Adaptive CUSUM")
    print("-" * 35)
    print(f"First alarm index: {new_alarm}")

    if new_alarm is not None:
        print(
            "First alarm date:",
            data["Date"].iloc[new_alarm].date()
        )
        print(
            f"CUSUM statistic: {new_stats[new_alarm]:.4f}"
        )

    if new_alarm is not None:
        print(
            "\nAlarm occurred after active warm-up:",
            np.sum(exposure[:new_alarm] > 0) >= WINDOW
        )

    print(
        "All historical statistics finite:",
        np.isfinite(new_stats).all()
    )

    print("\nHistorical Detector Diagnostics")
    print("-" * 35)

    print(
        f"Maximum CUSUM statistic: "
        f"{np.max(new_stats):.4f}"
    )

    print(
        f"Final CUSUM statistic: "
        f"{new_stats[-1]:.4f}"
    )

    print(
        f"Observations above threshold: "
        f"{np.sum(new_stats > THRESHOLD)}"
    )


def main():
    print("EDGEGUARD - Experiment 24")
    print("Original vs Historical Adaptive CUSUM")

    prices = load_historical_data(DATA_PATH)

    strategy = moving_average_strategy(
        prices,
        window=WINDOW,
    )

    train, validation, _ = chronological_split(strategy)

    evaluate("Training (2010-2018)", train)
    evaluate("Validation (2019-2022)", validation)


if __name__ == "__main__":
    main()
