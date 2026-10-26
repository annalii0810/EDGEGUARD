
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import rolling_mean_detector


# Experiment settings
N_SIMULATIONS = 500
CHANGE_POINT = 500
WINDOW = 50
SIGMA = 0.01
K_VALUES = [0, 1, 2, 3]

print("EDGEGUARD - Experiment 05")
print("Statistical Threshold Comparison")
print("--------------------------------")

for k in K_VALUES:

    # Calculate the alarm threshold
    threshold = -k * SIGMA / np.sqrt(WINDOW)

    false_alarms = 0
    successful = 0
    missed = 0
    delays = []

    for seed in range(N_SIMULATIONS):

        # Generate a strategy with deteriorating returns
        returns, _ = generate_returns(
            n=1000,
            change_point=CHANGE_POINT,
            mu_before=0.001,
            mu_after=-0.001,
            sigma=SIGMA,
            seed=seed
        )

        # Run the detector
        _, alarm = rolling_mean_detector(
            returns,
            window=WINDOW,
            threshold=threshold
        )

        if alarm is None:
            missed += 1

        elif alarm < CHANGE_POINT:
            false_alarms += 1

        else:
            successful += 1
            delays.append(alarm - CHANGE_POINT)

    print(f"\nThreshold k = {k}")
    print("-------------------------")
    print(f"Alarm threshold: {threshold:.6f}")
    print(f"False alarms: {false_alarms}")
    print(f"Successful detections: {successful}")
    print(f"Missed detections: {missed}")

    print(
        f"False-alarm proportion: "
        f"{false_alarms / N_SIMULATIONS:.4f}"
    )

    print(
        f"Successful-detection proportion: "
        f"{successful / N_SIMULATIONS:.4f}"
    )

    if delays:
        print(
            f"Average detection delay: "
            f"{np.mean(delays):.2f}"
        )
    else:
        print("Average detection delay: N/A")
