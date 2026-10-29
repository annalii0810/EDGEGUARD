
"""
EDGEGUARD - Experiment 34
Fair Detector Threshold Calibration

Exploratory Phase 5 research.
No historical test-period data is used.
"""

import numpy as np

from edgeguard.baseline_detectors import (
    rolling_mean_detector,
    standardized_cusum_detector,
)
from edgeguard.historical_detector import historical_adaptive_cusum


SEED = 2034
SIMULATIONS = 1000
OBSERVATIONS = 500

REFERENCE_MEAN = 0.0002299069244105812
VOLATILITY = 0.00715

TARGET_FALSE_ALARM = 0.05
WINDOW = 50


def edgeguard(returns):
    return historical_adaptive_cusum(
        returns,
        np.ones(len(returns)),
        window=WINDOW,
        target_mean=REFERENCE_MEAN,
        allowance=0.1,
        threshold=25.0,
        min_volatility=0.001,
    )


def rolling_mean(returns):
    return rolling_mean_detector(
        returns,
        window=WINDOW,
        target_mean=REFERENCE_MEAN,
        threshold=2.0,
        reference_volatility=VOLATILITY,
    )


def standard_cusum(returns):
    return standardized_cusum_detector(
        returns,
        target_mean=REFERENCE_MEAN,
        reference_volatility=VOLATILITY,
        allowance=0.1,
        threshold=25.0,
        warmup=WINDOW,
    )


DETECTORS = {
    "EDGEGUARD": edgeguard,
    "Rolling Mean": rolling_mean,
    "Standard CUSUM": standard_cusum,
}


def main():
    rng = np.random.default_rng(SEED)

    maximums = {
        name: []
        for name in DETECTORS
    }

    for _ in range(SIMULATIONS):
        returns = rng.normal(
            REFERENCE_MEAN,
            VOLATILITY,
            OBSERVATIONS,
        )

        for name, detector in DETECTORS.items():
            statistics, _ = detector(returns)

            maximums[name].append(
                float(np.max(statistics))
            )

    print("EDGEGUARD - Experiment 34")
    print("Fair Detector Calibration")
    print("=" * 65)

    print("Simulations:", SIMULATIONS)
    print("Monitoring horizon:", OBSERVATIONS)
    print("Target false-alarm rate:", TARGET_FALSE_ALARM)

    print("\nCalibration Results")
    print("-" * 65)

    for name, values in maximums.items():
        values = np.asarray(values)

        threshold = float(
            np.quantile(
                values,
                1 - TARGET_FALSE_ALARM,
            )
        )

        false_alarm_rate = float(
            np.mean(values > threshold)
        )

        print(f"\n{name}")
        print("Calibrated threshold:", round(threshold, 6))
        print(
            "Calibration false-alarm rate:",
            f"{false_alarm_rate:.2%}",
        )

    print("\nCalibration complete.")
    print(
        "Independent evaluation is required "
        "before comparing detector performance."
    )


if __name__ == "__main__":
    main()
