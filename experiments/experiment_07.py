
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import cusum_detector


# Experiment settings
N = 1000
N_CALIBRATION = 1000
TARGET_FALSE_ALARM = 0.05

TARGET_MEAN = 0.001
ALLOWANCE = 0.001
SIGMA = 0.01

# Store the maximum CUSUM statistic for each simulation
maximum_statistics = []

# Generate no-change calibration simulations
for seed in range(N_CALIBRATION):

    returns, _ = generate_returns(
        n=N,
        change_point=N,
        mu_before=TARGET_MEAN,
        mu_after=TARGET_MEAN,
        sigma=SIGMA,
        seed=seed
    )

    # Calculate CUSUM statistics with a very high threshold
    values, _ = cusum_detector(
        returns,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=1e100
    )

    maximum_statistics.append(np.max(values))

# Calibrate the alarm threshold
threshold = float(
    np.quantile(
        maximum_statistics,
        1 - TARGET_FALSE_ALARM
    )
)

print("EDGEGUARD - Experiment 07")
print("-------------------------")
print(f"Calibrated CUSUM threshold: {threshold:.6f}")



# Step 2: Independent no-change evaluation

N_TEST = 500
false_alarms = 0

for seed in range(10000, 10000 + N_TEST):

    # Generate returns without a regime change
    returns, _ = generate_returns(
        n=N,
        change_point=N,
        mu_before=TARGET_MEAN,
        mu_after=TARGET_MEAN,
        sigma=SIGMA,
        seed=seed
    )

    # Apply calibrated CUSUM detector
    _, alarm = cusum_detector(
        returns,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=threshold
    )

    if alarm is not None:
        false_alarms += 1

false_alarm_rate = false_alarms / N_TEST

print("\nIndependent No-Change Evaluation")
print("--------------------------------")
print(f"False alarms: {false_alarms}")
print(f"False-alarm proportion: {false_alarm_rate:.4f}")



# Step 3: Evaluate CUSUM during a regime change

CHANGE_POINT = 500

premature_alarms = 0
successful_detections = 0
missed_detections = 0
detection_delays = []

for seed in range(20000, 20000 + N_TEST):

    # Simulate a change from positive to negative returns
    returns, _ = generate_returns(
        n=N,
        change_point=CHANGE_POINT,
        mu_before=TARGET_MEAN,
        mu_after=-0.001,
        sigma=SIGMA,
        seed=seed
    )

    # Apply the calibrated CUSUM detector
    _, alarm = cusum_detector(
        returns,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=threshold
    )

    # Classify the first alarm
    if alarm is None:
        missed_detections += 1

    elif alarm < CHANGE_POINT:
        premature_alarms += 1

    else:
        successful_detections += 1
        detection_delays.append(alarm - CHANGE_POINT)


print("\nCUSUM Regime-Change Evaluation")
print("------------------------------")
print(f"Premature alarms: {premature_alarms}")
print(f"Successful detections: {successful_detections}")
print(f"Missed detections: {missed_detections}")

print(
    f"Successful-detection proportion: "
    f"{successful_detections / N_TEST:.4f}"
)

if detection_delays:
    print(
        f"Average detection delay: "
        f"{np.mean(detection_delays):.2f}"
    )
else:
    print("Average detection delay: N/A")
