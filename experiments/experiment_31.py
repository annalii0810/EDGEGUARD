
"""
EDGEGUARD - Experiment 31
Controlled Strategy Deterioration Simulation

Exploratory Phase 5 research.
No reserved historical test data is used.
"""

import numpy as np

from edgeguard.historical_detector import historical_adaptive_cusum


SEED = 2026
OBSERVATIONS = 500
CHANGE_POINT = 250
SIMULATIONS = 500

BASELINE_MEAN = 0.00023
DETERIORATED_MEAN = -0.003
VOLATILITY = 0.00715

WINDOW = 50
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001


def simulate_returns(rng, deteriorated=False):
    returns = rng.normal(
        BASELINE_MEAN,
        VOLATILITY,
        OBSERVATIONS,
    )

    if deteriorated:
        returns[CHANGE_POINT:] = rng.normal(
            DETERIORATED_MEAN,
            VOLATILITY,
            OBSERVATIONS - CHANGE_POINT,
        )

    return returns


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


def main():
    rng = np.random.default_rng(SEED)

    false_alarms = 0
    premature_alarms = 0
    detections = 0
    delays = []

    for _ in range(SIMULATIONS):
        normal = simulate_returns(
            rng, deteriorated=False
        )

        alarm = evaluate_path(normal)

        if alarm is not None:
            false_alarms += 1

        deteriorated = simulate_returns(
            rng, deteriorated=True
        )

        alarm = evaluate_path(deteriorated)

        if alarm is not None and alarm < CHANGE_POINT:
            premature_alarms += 1

        elif alarm is not None:
            detections += 1
            delays.append(alarm - CHANGE_POINT)

    print("EDGEGUARD - Experiment 31")
    print("Controlled Deterioration Simulation")
    print("=" * 55)

    print("Simulation paths:", SIMULATIONS)
    print("Observations per path:", OBSERVATIONS)
    print("Known change point:", CHANGE_POINT)
    print("Baseline mean:", BASELINE_MEAN)
    print("Deteriorated mean:", DETERIORATED_MEAN)
    print("Volatility:", VOLATILITY)

    print("\nResults")
    print("-" * 55)

    print(
        "False-alarm rate:",
        f"{false_alarms / SIMULATIONS:.2%}",
    )

    print(
        "Premature-alarm rate:",
        f"{premature_alarms / SIMULATIONS:.2%}",
    )

    print(
        "Detection rate:",
        f"{detections / SIMULATIONS:.2%}",
    )

    print(
        "Average detection delay:",
        np.mean(delays) if delays else "N/A",
    )

    print(
        "Median detection delay:",
        np.median(delays) if delays else "N/A",
    )

    print(
        "\nExploratory simulation only."
    )
    print(
        "Results do not establish real-market performance."
    )


if __name__ == "__main__":
    main()
