
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import cusum_detector


# Experiment settings
N = 1000
CHANGE_POINT = 500
N_SIMULATIONS = 500

VOLATILITIES = [0.005, 0.01, 0.02, 0.03]

TARGET_MEAN = 0.001
REFERENCE_ALLOWANCE = 0.1

# Initial threshold for testing only
THRESHOLD = 20.0


def normalized_cusum(returns, sigma, threshold):
    """CUSUM using known simulation volatility."""

    returns = np.asarray(returns, dtype=float)

    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Sigma must be positive and finite")

    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("Returns must be a nonempty 1D array")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be finite")

    normalized = (TARGET_MEAN - returns) / sigma

    # The existing downward CUSUM function expects
    # observed returns, so negate normalized deviations.
    values, alarm = cusum_detector(
        -normalized,
        target_mean=0.0,
        allowance=REFERENCE_ALLOWANCE,
        threshold=threshold
    )

    return values, alarm


print("EDGEGUARD - Experiment 09")
print("Volatility-Normalized CUSUM")
print("--------------------------------")

for sigma in VOLATILITIES:

    premature = 0
    successful = 0
    missed = 0
    delays = []

    for seed in range(40000, 40000 + N_SIMULATIONS):

        returns, _ = generate_returns(
            n=N,
            change_point=CHANGE_POINT,
            mu_before=0.001,
            mu_after=-0.001,
            sigma=sigma,
            seed=seed
        )

        _, alarm = normalized_cusum(
            returns,
            sigma=sigma,
            threshold=THRESHOLD
        )

        if alarm is None:
            missed += 1

        elif alarm < CHANGE_POINT:
            premature += 1

        else:
            successful += 1
            delays.append(alarm - CHANGE_POINT)

    print(f"\nVolatility: {sigma:.3f}")
    print(f"Premature alarms: {premature / N_SIMULATIONS:.1%}")
    print(f"Successful detections: {successful / N_SIMULATIONS:.1%}")
    print(f"Missed detections: {missed / N_SIMULATIONS:.1%}")

    if delays:
        print(f"Average detection delay: {np.mean(delays):.2f}")
    else:
        print("Average detection delay: N/A")
