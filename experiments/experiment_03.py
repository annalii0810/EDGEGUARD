
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import rolling_mean_detector


# Experiment settings
N_SIMULATIONS = 500
CHANGE_POINT = 500
WINDOWS = [10, 50, 100, 200]

# Compare each rolling-window size
for window in WINDOWS:

    false_alarms = 0
    successful = 0
    missed = 0
    delays = []

    # Run 500 simulations for each window
    for seed in range(N_SIMULATIONS):

        returns, _ = generate_returns(
            n=1000,
            change_point=CHANGE_POINT,
            mu_before=0.001,
            mu_after=-0.001,
            sigma=0.01,
            seed=seed
        )

        _, alarm = rolling_mean_detector(
            returns,
            window=window,
            threshold=0.0
        )

        if alarm is None:
            missed += 1

        elif alarm < CHANGE_POINT:
            false_alarms += 1

        else:
            successful += 1
            delays.append(alarm - CHANGE_POINT)

    # Calculate summary statistics
    false_alarm_rate = false_alarms / N_SIMULATIONS
    detection_rate = successful / N_SIMULATIONS

    avg_delay = (
        float(np.mean(delays))
        if delays else float("nan")
    )

    print(f"\nRolling Window: {window}")
    print("-------------------------")
    print(f"False alarms: {false_alarms}")
    print(f"Successful detections: {successful}")
    print(f"Missed detections: {missed}")
    print(f"False-alarm proportion: {false_alarm_rate:.4f}")
    print(f"Successful-detection proportion: {detection_rate:.4f}")
    print(f"Average delay among detections: {avg_delay:.2f}")
