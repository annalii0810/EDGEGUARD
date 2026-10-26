
from edgeguard.simulator import generate_returns
from edgeguard.detectors import rolling_mean_detector


# Experiment settings
N_SIMULATIONS = 500
WINDOWS = [10, 50, 100, 200]

print("EDGEGUARD - Experiment 04")
print("No-Change False-Alarm Evaluation")
print("--------------------------------")

# Test each rolling-window size
for window in WINDOWS:

    false_alarms = 0

    # Generate 500 independent simulations
    for seed in range(N_SIMULATIONS):

        returns, _ = generate_returns(
            n=1000,
            change_point=1000,
            mu_before=0.001,
            mu_after=-0.001,
            sigma=0.01,
            seed=seed
        )

        # Apply the rolling-mean detector
        _, alarm = rolling_mean_detector(
            returns,
            window=window,
            threshold=0.0
        )

        # Any alarm is false because no change occurred
        if alarm is not None:
            false_alarms += 1

    false_alarm_rate = false_alarms / N_SIMULATIONS

    print(f"\nRolling Window: {window}")
    print("-------------------------")
    print(f"False alarms: {false_alarms}")
    print(f"False-alarm proportion: {false_alarm_rate:.4f}")
