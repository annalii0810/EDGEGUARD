
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import rolling_mean_detector


# Experiment settings
CHANGE_POINT = 500
WINDOW = 50
N_SIMULATIONS = 500

# Counters for detector outcomes
false_alarms = 0
successful_detections = 0
missed_detections = 0

# Store detection delays
delays = []

# Repeat the experiment 500 times
for seed in range(N_SIMULATIONS):

    # Generate a new simulated return sequence
    returns, _ = generate_returns(
        n=1000,
        change_point=CHANGE_POINT,
        mu_before=0.001,
        mu_after=-0.001,
        sigma=0.01,
        seed=seed
    )

    # Run the rolling-mean detector
    _, alarm = rolling_mean_detector(
        returns,
        window=WINDOW,
        threshold=0.0
    )

    # Classify the detector's first alarm
    if alarm is None:
        missed_detections += 1

    elif alarm < CHANGE_POINT:
        false_alarms += 1

    else:
        successful_detections += 1
        delays.append(alarm - CHANGE_POINT)


# Display results
print("EDGEGUARD - Experiment 02")
print("-------------------------")
print(f"Total simulations: {N_SIMULATIONS}")
print(f"Rolling window: {WINDOW}")
print()

print(f"False alarms: {false_alarms}")
print(f"Successful detections: {successful_detections}")
print(f"Missed detections: {missed_detections}")
print()

print(
    "False-alarm proportion:",
    round(false_alarms / N_SIMULATIONS, 4)
)

print(
    "Successful-detection proportion:",
    round(successful_detections / N_SIMULATIONS, 4)
)

if delays:
    print(
        "Average detection delay:",
        round(float(np.mean(delays)), 2)
    )
else:
    print("Average detection delay: N/A")
