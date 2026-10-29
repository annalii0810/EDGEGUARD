
"""
EDGEGUARD - Experiment 35
Independent Synthetic Detector Comparison

Exploratory research only.
No historical test-period data is used.
"""

import numpy as np

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.baseline_detectors import (
    rolling_mean_detector,
    standardized_cusum_detector,
)
from edgeguard.statistical_evaluation import wilson_interval


SEED = 2035
SIMULATIONS = 1000
OBSERVATIONS = 500
CHANGE_POINT = 250

REFERENCE_MEAN = 0.0002299069244105812
DETERIORATED_MEAN = -0.003
VOLATILITY = 0.00715

WINDOW = 50

# Thresholds calibrated in Experiment 34.
# These are for simulation comparisons only.
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

    return standardized_cusum_detector(
        returns,
        target_mean=REFERENCE_MEAN,
        reference_volatility=VOLATILITY,
        allowance=0.1,
        threshold=THRESHOLDS[name],
        warmup=WINDOW,
    )[1]


def main():
    rng = np.random.default_rng(SEED)

    names = list(THRESHOLDS)

    results = {
        name: {
            "false": 0,
            "premature": 0,
            "detected": 0,
            "missed": 0,
            "delays": [],
        }
        for name in names
    }

    for _ in range(SIMULATIONS):
        # Identical no-change path for all detectors.
        normal = rng.normal(
            REFERENCE_MEAN,
            VOLATILITY,
            OBSERVATIONS,
        )

        # Separate deteriorated path.
        changed = rng.normal(
            REFERENCE_MEAN,
            VOLATILITY,
            OBSERVATIONS,
        )

        changed[CHANGE_POINT:] = rng.normal(
            DETERIORATED_MEAN,
            VOLATILITY,
            OBSERVATIONS - CHANGE_POINT,
        )

        for name in names:
            result = results[name]

            normal_alarm = detect(name, normal)

            if normal_alarm is not None:
                result["false"] += 1

            alarm = detect(name, changed)

            if alarm is None:
                result["missed"] += 1
            elif alarm < CHANGE_POINT:
                result["premature"] += 1
            else:
                result["detected"] += 1
                result["delays"].append(
                    alarm - CHANGE_POINT
                )

    print("EDGEGUARD - Experiment 35")
    print("Independent Detector Comparison")
    print("=" * 85)

    print("Independent paths:", SIMULATIONS)
    print("Observations:", OBSERVATIONS)
    print("Change point:", CHANGE_POINT)
    print("Deteriorated mean:", DETERIORATED_MEAN)

    for name in names:
        r = results[name]

        false_rate = r["false"] / SIMULATIONS
        detection_rate = r["detected"] / SIMULATIONS

        false_ci = wilson_interval(
            r["false"], SIMULATIONS
        )
        detection_ci = wilson_interval(
            r["detected"], SIMULATIONS
        )

        print(f"\n{name}")
        print("-" * 50)
        print("Threshold:", THRESHOLDS[name])
        print(
            f"False alarms: {false_rate:.2%} "
            f"(95% CI: {false_ci[0]:.2%} "
            f"to {false_ci[1]:.2%})"
        )
        print(
            "Premature alarms:",
            f"{r['premature'] / SIMULATIONS:.2%}",
        )
        print(
            f"Post-change detections: {detection_rate:.2%} "
            f"(95% CI: {detection_ci[0]:.2%} "
            f"to {detection_ci[1]:.2%})"
        )
        print(
            "Missed detections:",
            f"{r['missed'] / SIMULATIONS:.2%}",
        )
        print(
            "Average detection delay:",
            (
                f"{np.mean(r['delays']):.1f}"
                if r["delays"] else "N/A"
            ),
        )

    print("\nIndependent synthetic comparison complete.")
    print("Historical Phase 4 results remain unchanged.")


if __name__ == "__main__":
    main()
