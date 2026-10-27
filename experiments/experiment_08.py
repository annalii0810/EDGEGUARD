
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import rolling_mean_detector, cusum_detector


# Experiment settings
N = 1000
CHANGE_POINT = 500
N_SIMULATIONS = 500

VOLATILITIES = [0.005, 0.01, 0.02, 0.03]

# Previously calibrated thresholds
ROLLING_THRESHOLD = -0.003852
CUSUM_THRESHOLD = 0.273950

WINDOW = 50
TARGET_MEAN = 0.001
ALLOWANCE = 0.001


def evaluate_detector(alarms):
    """Summarize first-alarm outcomes."""
    premature = 0
    successful = 0
    missed = 0
    delays = []

    for alarm in alarms:
        if alarm is None:
            missed += 1
        elif alarm < CHANGE_POINT:
            premature += 1
        else:
            successful += 1
            delays.append(alarm - CHANGE_POINT)

    return {
        "premature": premature / len(alarms),
        "successful": successful / len(alarms),
        "missed": missed / len(alarms),
        "average_delay": (
            float(np.mean(delays)) if delays else float("nan")
        ),
    }


print("EDGEGUARD - Experiment 08")
print("Volatility Robustness Evaluation")
print("--------------------------------")

for sigma in VOLATILITIES:
    rolling_alarms = []
    cusum_alarms = []

    for seed in range(30000, 30000 + N_SIMULATIONS):
        returns, _ = generate_returns(
            n=N,
            change_point=CHANGE_POINT,
            mu_before=0.001,
            mu_after=-0.001,
            sigma=sigma,
            seed=seed,
        )

        _, rolling_alarm = rolling_mean_detector(
            returns,
            window=WINDOW,
            threshold=ROLLING_THRESHOLD,
        )

        _, cusum_alarm = cusum_detector(
            returns,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=CUSUM_THRESHOLD,
        )

        rolling_alarms.append(rolling_alarm)
        cusum_alarms.append(cusum_alarm)

    rolling = evaluate_detector(rolling_alarms)
    cusum = evaluate_detector(cusum_alarms)

    print(f"\nVolatility: {sigma:.3f}")

    for name, result in [
        ("Rolling Mean", rolling),
        ("CUSUM", cusum),
    ]:
        print(f"  {name}")
        print(f"    Premature alarms: {result['premature']:.1%}")
        print(f"    Successful detections: {result['successful']:.1%}")
        print(f"    Missed detections: {result['missed']:.1%}")
        print(f"    Average delay: {result['average_delay']:.2f}")
