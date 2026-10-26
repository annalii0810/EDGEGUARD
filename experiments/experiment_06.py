
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import rolling_mean_detector


# Experiment settings
WINDOW = 50
N = 1000
N_CALIBRATION = 1000
N_TEST = 500
TARGET_FALSE_ALARM = 0.05

# Step 1: Collect minimum rolling means
# from no-change calibration simulations.
minimum_means = []

for seed in range(N_CALIBRATION):
    returns, _ = generate_returns(
        n=N,
        change_point=N,
        mu_before=0.001,
        mu_after=0.001,
        sigma=0.01,
        seed=seed,
    )

    rolling_means, _ = rolling_mean_detector(
        returns,
        window=WINDOW,
        threshold=-1e100,
    )

    minimum_means.append(
        np.nanmin(rolling_means)
    )

# Step 2: Choose an empirical threshold.
# Use the 5th percentile of minimum rolling means.
threshold = float(
    np.quantile(
        minimum_means,
        TARGET_FALSE_ALARM
    )
)

print("EDGEGUARD - Experiment 06")
print("-------------------------")
print(f"Calibrated threshold: {threshold:.6f}")

# Step 3: Evaluate on independent no-change simulations
false_alarms = 0

for seed in range(10000, 10000 + N_TEST):
    returns, _ = generate_returns(
        n=N,
        change_point=N,
        mu_before=0.001,
        mu_after=0.001,
        sigma=0.01,
        seed=seed,
    )

    _, alarm = rolling_mean_detector(
        returns,
        window=WINDOW,
        threshold=threshold,
    )

    if alarm is not None:
        false_alarms += 1

print(
    "Out-of-sample false-alarm proportion:",
    round(false_alarms / N_TEST, 4)
)



# Step 4: Test detection after a real regime change
change_point = 500

premature_alarms = 0
successful_detections = 0
missed_detections = 0
detection_delays = []

for seed in range(20000, 20000 + N_TEST):
    returns, _ = generate_returns(
        n=N,
        change_point=change_point,
        mu_before=0.001,
        mu_after=-0.001,
        sigma=0.01,
        seed=seed,
    )

    _, alarm = rolling_mean_detector(
        returns,
        window=WINDOW,
        threshold=threshold,
    )

    if alarm is None:
        missed_detections += 1

    elif alarm < change_point:
        premature_alarms += 1

    else:
        successful_detections += 1
        detection_delays.append(alarm - change_point)

print("\nRegime-Change Evaluation")
print("-------------------------")
print(f"Premature alarms: {premature_alarms}")
print(f"Successful detections: {successful_detections}")
print(f"Missed detections: {missed_detections}")

print(
    f"Successful-detection proportion: "
    f"{successful_detections / N_TEST:.4f}"
)

if detection_delays:
    print(
        f"Average delay among detections: "
        f"{np.mean(detection_delays):.2f}"
    )
else:
    print("Average delay among detections: N/A")
