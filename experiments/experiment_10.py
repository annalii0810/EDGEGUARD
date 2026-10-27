
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import cusum_detector


# Experiment settings
N = 1000
N_CALIBRATION = 1000
TARGET_FALSE_ALARM = 0.05

TARGET_MEAN = 0.001
REFERENCE_ALLOWANCE = 0.1
SIGMA = 0.01


def normalized_cusum(returns, sigma, threshold):
    """CUSUM normalized by known simulation volatility."""

    returns = np.asarray(returns, dtype=float)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite")

    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("Returns must be a nonempty 1D array")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be finite")

    normalized = (TARGET_MEAN - returns) / sigma

    values, alarm = cusum_detector(
        -normalized,
        target_mean=0.0,
        allowance=REFERENCE_ALLOWANCE,
        threshold=threshold,
    )

    return values, alarm


# Step 1: Collect maximum CUSUM statistics
maximum_statistics = []

for seed in range(N_CALIBRATION):

    returns, _ = generate_returns(
        n=N,
        change_point=N,
        mu_before=TARGET_MEAN,
        mu_after=TARGET_MEAN,
        sigma=SIGMA,
        seed=seed,
    )

    values, _ = normalized_cusum(
        returns,
        sigma=SIGMA,
        threshold=1e100,
    )

    maximum_statistics.append(float(np.max(values)))


# Step 2: Calibrate the threshold
threshold = float(
    np.quantile(
        maximum_statistics,
        1 - TARGET_FALSE_ALARM,
    )
)

print("EDGEGUARD - Experiment 10")
print("Normalized CUSUM Calibration")
print("--------------------------------")
print(f"Calibrated threshold: {threshold:.6f}")



# Step 3: Independent no-change testing

N_TEST = 500
VOLATILITIES = [0.005, 0.01, 0.02, 0.03]

print("\nIndependent No-Change Evaluation")
print("--------------------------------")

for sigma in VOLATILITIES:

    false_alarms = 0

    for seed in range(10000, 10000 + N_TEST):

        # Generate returns with no deterioration
        returns, _ = generate_returns(
            n=N,
            change_point=N,
            mu_before=TARGET_MEAN,
            mu_after=TARGET_MEAN,
            sigma=sigma,
            seed=seed,
        )

        # Apply the calibrated normalized CUSUM
        _, alarm = normalized_cusum(
            returns,
            sigma=sigma,
            threshold=threshold,
        )

        if alarm is not None:
            false_alarms += 1

    false_alarm_rate = false_alarms / N_TEST

    print(f"\nVolatility: {sigma:.3f}")
    print(f"False alarms: {false_alarms}")
    print(f"False-alarm proportion: {false_alarm_rate:.4f}")



# Step 4: Evaluate calibrated CUSUM under regime changes

CHANGE_POINT = 500

print("\nCalibrated Regime-Change Evaluation")
print("-----------------------------------")

for sigma in VOLATILITIES:

    premature = 0
    successful = 0
    missed = 0
    delays = []

    for seed in range(20000, 20000 + N_TEST):

        # Generate a strategy with deteriorating returns
        returns, _ = generate_returns(
            n=N,
            change_point=CHANGE_POINT,
            mu_before=TARGET_MEAN,
            mu_after=-0.001,
            sigma=sigma,
            seed=seed,
        )

        # Apply the calibrated normalized detector
        _, alarm = normalized_cusum(
            returns,
            sigma=sigma,
            threshold=threshold,
        )

        if alarm is None:
            missed += 1

        elif alarm < CHANGE_POINT:
            premature += 1

        else:
            successful += 1
            delays.append(alarm - CHANGE_POINT)

    print(f"\nVolatility: {sigma:.3f}")
    print(f"Premature alarms: {premature / N_TEST:.1%}")
    print(f"Successful detections: {successful / N_TEST:.1%}")
    print(f"Missed detections: {missed / N_TEST:.1%}")

    if delays:
        print(
            f"Average delay among detections: "
            f"{np.mean(delays):.2f}"
        )
    else:
        print("Average delay among detections: N/A")
