
"""
EDGEGUARD - Experiment 36
Paired Statistical Detector Comparison

Exploratory synthetic evaluation.
No historical test-period data is used.
"""

import numpy as np

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.baseline_detectors import (
    rolling_mean_detector,
    standardized_cusum_detector,
)

SEED = 2036
SIMULATIONS = 1000
OBSERVATIONS = 500
CHANGE_POINT = 250
DEADLINES = (50, 100)

REFERENCE_MEAN = 0.0002299069244105812
DETERIORATED_MEAN = -0.003
VOLATILITY = 0.00715
WINDOW = 50

THRESHOLDS = {
    "EDGEGUARD": 24.408925,
    "Rolling Mean": 3.170221,
    "Standard CUSUM": 23.387877,
}


def detect(name, returns):
    if name == "EDGEGUARD":
        return historical_adaptive_cusum(
            returns,
            np.ones(len(returns)),
            window=WINDOW,
            target_mean=REFERENCE_MEAN,
            allowance=0.1,
            threshold=THRESHOLDS[name],
            min_volatility=0.001,
        )[1]

    if name == "Rolling Mean":
        return rolling_mean_detector(
            returns,
            window=WINDOW,
            target_mean=REFERENCE_MEAN,
            threshold=THRESHOLDS[name],
            reference_volatility=VOLATILITY,
        )[1]

    if name == "Standard CUSUM":
        return standardized_cusum_detector(
            returns,
            target_mean=REFERENCE_MEAN,
            reference_volatility=VOLATILITY,
            allowance=0.1,
            threshold=THRESHOLDS[name],
            warmup=WINDOW,
        )[1]

    raise ValueError("Unknown detector")


def bootstrap_paired_ci(differences, seed, repetitions=2000):
    """Paired bootstrap confidence interval."""
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(differences)

    estimates = np.empty(repetitions)

    for i in range(repetitions):
        indices = rng.integers(0, n, size=n)
        estimates[i] = differences[indices].mean()

    return np.quantile(estimates, [0.025, 0.975])


def main():
    rng = np.random.default_rng(SEED)
    names = list(THRESHOLDS)

    alarms = {
        name: []
        for name in names
    }

    for _ in range(SIMULATIONS):
        returns = rng.normal(
            REFERENCE_MEAN,
            VOLATILITY,
            OBSERVATIONS,
        )

        returns[CHANGE_POINT:] = rng.normal(
            DETERIORATED_MEAN,
            VOLATILITY,
            OBSERVATIONS - CHANGE_POINT,
        )

        for name in names:
            alarms[name].append(
                detect(name, returns)
            )

    print("EDGEGUARD - Experiment 36")
    print("Paired Statistical Comparison")
    print("=" * 65)

    for deadline in DEADLINES:
        print(
            f"\nDetection within {deadline} observations"
        )
        print("-" * 65)

        outcomes = {}

        for name in names:
            outcomes[name] = np.array([
                alarm is not None
                and CHANGE_POINT <= alarm <= CHANGE_POINT + deadline
                for alarm in alarms[name]
            ], dtype=int)

            print(
                f"{name}: {outcomes[name].mean():.2%}"
            )

        for baseline in names[1:]:
            differences = (
                outcomes["EDGEGUARD"]
                - outcomes[baseline]
            )

            low, high = bootstrap_paired_ci(
                differences,
                seed=SEED + deadline + names.index(baseline),
            )

            edge_only = int(np.sum(
                (outcomes["EDGEGUARD"] == 1)
                & (outcomes[baseline] == 0)
            ))

            baseline_only = int(np.sum(
                (outcomes["EDGEGUARD"] == 0)
                & (outcomes[baseline] == 1)
            ))

            print(f"\nEDGEGUARD vs {baseline}")
            print("EDGEGUARD-only detections:", edge_only)
            print("Baseline-only detections:", baseline_only)
            print(
                "Paired detection difference:",
                f"{differences.mean():+.2%}",
            )
            print(
                "95% bootstrap CI:",
                f"[{low:+.2%}, {high:+.2%}]",
            )

    print("\nPaired delay comparisons")
    print("-" * 65)

    for baseline in names[1:]:
        paired_delays = []

        for edge_alarm, base_alarm in zip(
            alarms["EDGEGUARD"],
            alarms[baseline],
        ):
            if (
                edge_alarm is not None
                and base_alarm is not None
                and edge_alarm >= CHANGE_POINT
                and base_alarm >= CHANGE_POINT
            ):
                paired_delays.append(
                    edge_alarm - base_alarm
                )

        if paired_delays:
            differences = np.array(paired_delays)

            low, high = bootstrap_paired_ci(
                differences,
                seed=SEED + 100 + names.index(baseline),
            )

            print(f"\nEDGEGUARD vs {baseline}")
            print("Paired detections:", len(differences))
            print(
                "Mean delay difference:",
                f"{differences.mean():+.2f}",
            )
            print(
                "95% bootstrap CI:",
                f"[{low:+.2f}, {high:+.2f}]",
            )

    print("\nInterpretation:")
    print("Negative delay difference means EDGEGUARD is faster.")
    print("Positive detection difference favors EDGEGUARD.")
    print("Confidence intervals are exploratory.")
    print("Historical Phase 4 results remain unchanged.")


if __name__ == "__main__":
    main()
