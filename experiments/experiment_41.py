
"""
EDGEGUARD - Experiment 41
False-Alarm Sensitivity to Volatility Misspecification

Exploratory synthetic research only.
No historical test-period data is used.
"""

import numpy as np

from edgeguard.historical_detector import historical_adaptive_cusum
from edgeguard.baseline_detectors import standardized_cusum_detector
from edgeguard.statistical_evaluation import wilson_interval


SEED = 2041
SIMULATIONS = 500
OBSERVATIONS = 500

REFERENCE_MEAN = 0.0002299069244105812
TRUE_VOLATILITY = 0.00715

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001

EDGE_THRESHOLD = 24.408925
STANDARD_THRESHOLD = 23.387877

MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def edgeguard_alarm(returns):
    return historical_adaptive_cusum(
        returns,
        np.ones(len(returns)),
        window=WINDOW,
        target_mean=REFERENCE_MEAN,
        allowance=ALLOWANCE,
        threshold=EDGE_THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )[1]


def standard_alarm(returns, assumed_volatility):
    return standardized_cusum_detector(
        returns,
        target_mean=REFERENCE_MEAN,
        reference_volatility=assumed_volatility,
        allowance=ALLOWANCE,
        threshold=STANDARD_THRESHOLD,
        warmup=WINDOW,
    )[1]


def main():
    rng = np.random.default_rng(SEED)

    # Generate once, then reuse the same paths
    # across every detector configuration.
    paths = rng.normal(
        REFERENCE_MEAN,
        TRUE_VOLATILITY,
        size=(SIMULATIONS, OBSERVATIONS),
    )

    edge_alarms = sum(
        edgeguard_alarm(path) is not None
        for path in paths
    )

    edge_low, edge_high = wilson_interval(
        edge_alarms, SIMULATIONS
    )

    print("EDGEGUARD - Experiment 41")
    print("False-Alarm Sensitivity Analysis")
    print("=" * 80)

    print("Paths:", SIMULATIONS)
    print("Observations per path:", OBSERVATIONS)
    print("True volatility:", TRUE_VOLATILITY)

    print("\nEDGEGUARD")
    print("False alarms:", edge_alarms)
    print("False-alarm rate:", f"{edge_alarms / SIMULATIONS:.2%}")
    print(
        "95% Wilson CI:",
        f"[{edge_low:.2%}, {edge_high:.2%}]"
    )

    print("\nStandard CUSUM")
    print("-" * 80)
    print(
        f"{'Vol Mult':>10} "
        f"{'False Alarms':>14} "
        f"{'Rate':>12} "
        f"{'95% CI':>24}"
    )

    for multiplier in MULTIPLIERS:
        assumed_volatility = TRUE_VOLATILITY * multiplier

        alarms = sum(
            standard_alarm(path, assumed_volatility) is not None
            for path in paths
        )

        low, high = wilson_interval(alarms, SIMULATIONS)

        print(
            f"{multiplier:>10.2f} "
            f"{alarms:>14} "
            f"{alarms / SIMULATIONS:>12.2%} "
            f"{f'{low:.2%} to {high:.2%}':>24}"
        )

    print("\nExploratory sensitivity analysis complete.")
    print("Historical Phase 4 results unchanged.")


if __name__ == "__main__":
    main()
