
"""
EDGEGUARD - Experiment 33
Volatility Regime Sensitivity

Exploratory synthetic research.
No reserved historical test data is used.
"""

import numpy as np

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.statistical_evaluation import wilson_interval


SEED = 2033
SIMULATIONS = 500
OBSERVATIONS = 500
CHANGE_POINT = 250

BASELINE_MEAN = 0.00023
DETERIORATED_MEAN = -0.003

VOLATILITIES = [0.003, 0.005, 0.00715, 0.01, 0.015, 0.02]

WINDOW = 50
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001


def detect(returns):
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


def evaluate(volatility, rng):
    false_alarms = 0
    premature = 0
    detected = 0
    delays = []

    for _ in range(SIMULATIONS):
        normal = rng.normal(
            BASELINE_MEAN,
            volatility,
            OBSERVATIONS,
        )

        if detect(normal) is not None:
            false_alarms += 1

        changed = rng.normal(
            BASELINE_MEAN,
            volatility,
            OBSERVATIONS,
        )

        changed[CHANGE_POINT:] = rng.normal(
            DETERIORATED_MEAN,
            volatility,
            OBSERVATIONS - CHANGE_POINT,
        )

        alarm = detect(changed)

        if alarm is not None and alarm < CHANGE_POINT:
            premature += 1
        elif alarm is not None:
            detected += 1
            delays.append(alarm - CHANGE_POINT)

    false_ci = wilson_interval(false_alarms, SIMULATIONS)
    detection_ci = wilson_interval(detected, SIMULATIONS)

    return {
        "volatility": volatility,
        "false_alarm": false_alarms / SIMULATIONS,
        "false_ci": false_ci,
        "premature": premature / SIMULATIONS,
        "detection": detected / SIMULATIONS,
        "detection_ci": detection_ci,
        "delay": float(np.mean(delays)) if delays else np.nan,
    }


def main():
    rng = np.random.default_rng(SEED)

    print("EDGEGUARD - Experiment 33")
    print("Volatility Regime Sensitivity")
    print("=" * 95)

    print("Simulations per regime:", SIMULATIONS)
    print("Known change point:", CHANGE_POINT)
    print("Frozen threshold:", THRESHOLD)

    print("\nResults")
    print("=" * 95)

    print(
        f"{'Volatility':>12} "
        f"{'False Alarm':>12} "
        f"{'Premature':>12} "
        f"{'Detection':>12} "
        f"{'95% Detection CI':>22} "
        f"{'Avg Delay':>12}"
    )

    for volatility in VOLATILITIES:
        result = evaluate(volatility, rng)

        low, high = result["detection_ci"]

        print(
            f"{volatility:>12.5f} "
            f"{result['false_alarm']:>12.1%} "
            f"{result['premature']:>12.1%} "
            f"{result['detection']:>12.1%} "
            f"{f'{low:.1%} to {high:.1%}':>22} "
            f"{result['delay']:>12.1f}"
        )

    print("\nExploratory simulation only.")
    print("Frozen detector parameters unchanged.")


if __name__ == "__main__":
    main()
