
"""
EDGEGUARD - Experiment 32
Detection Sensitivity Across Deterioration Strengths

Exploratory simulation. No historical test data.
"""

import numpy as np

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.statistical_evaluation import wilson_interval


SEED = 2032
SIMULATIONS = 500
OBSERVATIONS = 500
CHANGE_POINT = 250

BASELINE_MEAN = 0.00023
VOLATILITY = 0.00715

DETERIORATED_MEANS = [
    0.00023,
    -0.0005,
    -0.001,
    -0.002,
    -0.003,
    -0.005,
]

WINDOW = 50
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001


def evaluate_path(returns):
    exposure = np.ones(len(returns))

    _, alarm = historical_adaptive_cusum(
        returns,
        exposure,
        window=WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    return alarm


def evaluate_scenario(mean, rng):
    premature = 0
    detected = 0
    missed = 0
    delays = []

    for _ in range(SIMULATIONS):
        returns = rng.normal(
            BASELINE_MEAN,
            VOLATILITY,
            OBSERVATIONS,
        )

        returns[CHANGE_POINT:] = rng.normal(
            mean,
            VOLATILITY,
            OBSERVATIONS - CHANGE_POINT,
        )

        alarm = evaluate_path(returns)

        if alarm is None:
            missed += 1
        elif alarm < CHANGE_POINT:
            premature += 1
        else:
            detected += 1
            delays.append(alarm - CHANGE_POINT)

    low, high = wilson_interval(
        detected, SIMULATIONS
    )

    return {
        "mean": mean,
        "detected": detected,
        "premature": premature,
        "missed": missed,
        "detection_rate": detected / SIMULATIONS,
        "ci_low": low,
        "ci_high": high,
        "average_delay": (
            float(np.mean(delays)) if delays else np.nan
        ),
    }


def main():
    rng = np.random.default_rng(SEED)

    print("EDGEGUARD - Experiment 32")
    print("Detection Sensitivity Analysis")
    print("=" * 90)

    print("Simulations per scenario:", SIMULATIONS)
    print("Change point:", CHANGE_POINT)
    print("Detector threshold:", THRESHOLD)

    print("\nResults")
    print("=" * 90)

    print(
        f"{'Mean':>10} "
        f"{'Detection':>12} "
        f"{'95% CI':>22} "
        f"{'Premature':>12} "
        f"{'Missed':>10} "
        f"{'Avg Delay':>12}"
    )

    for mean in DETERIORATED_MEANS:
        result = evaluate_scenario(mean, rng)

        ci = (
            f"{result['ci_low']:.1%}–"
            f"{result['ci_high']:.1%}"
        )

        print(
            f"{mean:>10.5f} "
            f"{result['detection_rate']:>12.1%} "
            f"{ci:>22} "
            f"{result['premature']:>12} "
            f"{result['missed']:>10} "
            f"{result['average_delay']:>12.1f}"
        )

    print("\nExploratory synthetic results only.")
    print("The frozen detector parameters were unchanged.")


if __name__ == "__main__":
    main()
