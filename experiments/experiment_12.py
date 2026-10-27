
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import adaptive_cusum_detector


# Experiment 12: Adaptive CUSUM calibration
N = 1000
N_CALIBRATION = 1000

TARGET_MEAN = 0.001
SIGMA = 0.01
WINDOW = 50
ALLOWANCE = 0.1
TARGET_FALSE_ALARM = 0.05

VOLATILITY_FLOORS = [1e-6, 0.0005, 0.001]


print("EDGEGUARD - Experiment 12")
print("Adaptive CUSUM Threshold Calibration")
print("------------------------------------")


# Generate calibration sequences once.
# Every candidate uses the same sequences for fairness.
calibration_returns = []

for seed in range(N_CALIBRATION):
    returns, _ = generate_returns(
        n=N,
        change_point=N,
        mu_before=TARGET_MEAN,
        mu_after=TARGET_MEAN,
        sigma=SIGMA,
        seed=seed,
    )
    calibration_returns.append(returns)


# Calibrate each volatility floor separately.
calibrated_thresholds = {}

for floor in VOLATILITY_FLOORS:
    maximum_statistics = []

    for returns in calibration_returns:
        values, _ = adaptive_cusum_detector(
            returns,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=1e100,
            min_volatility=floor,
        )

        maximum_statistics.append(
            float(np.max(values))
        )

    threshold = float(
        np.quantile(
            maximum_statistics,
            1 - TARGET_FALSE_ALARM,
        )
    )

    calibrated_thresholds[floor] = threshold

    print(f"\nVolatility floor: {floor:.6f}")
    print(f"Calibrated threshold: {threshold:.6f}")



# Step 3: Independent false-alarm evaluation

N_TEST = 500
VOLATILITIES = [0.005, 0.01, 0.02, 0.03]

print("\nIndependent False-Alarm Evaluation")
print("----------------------------------")

for floor in VOLATILITY_FLOORS:

    threshold = calibrated_thresholds[floor]

    print(f"\nVolatility floor: {floor:.6f}")
    print(f"Calibrated threshold: {threshold:.6f}")

    for sigma in VOLATILITIES:

        false_alarms = 0

        for seed in range(10000, 10000 + N_TEST):

            # No regime change occurs
            returns, _ = generate_returns(
                n=N,
                change_point=N,
                mu_before=TARGET_MEAN,
                mu_after=TARGET_MEAN,
                sigma=sigma,
                seed=seed,
            )

            _, alarm = adaptive_cusum_detector(
                returns,
                window=WINDOW,
                target_mean=TARGET_MEAN,
                allowance=ALLOWANCE,
                threshold=threshold,
                min_volatility=floor,
            )

            if alarm is not None:
                false_alarms += 1

        false_alarm_rate = false_alarms / N_TEST

        print(f"\nVolatility: {sigma:.3f}")
        print(f"False alarms: {false_alarms}")
        print(f"False-alarm proportion: {false_alarm_rate:.4f}")



# Step 4: Independent regime-change evaluation

CHANGE_POINT = 500

print("\nAdaptive CUSUM Regime-Change Evaluation")
print("---------------------------------------")

for floor in VOLATILITY_FLOORS:

    threshold = calibrated_thresholds[floor]

    print(f"\nVolatility floor: {floor:.6f}")

    for sigma in VOLATILITIES:

        premature = 0
        successful = 0
        missed = 0
        delays = []

        for seed in range(20000, 20000 + N_TEST):

            returns, _ = generate_returns(
                n=N,
                change_point=CHANGE_POINT,
                mu_before=TARGET_MEAN,
                mu_after=-0.001,
                sigma=sigma,
                seed=seed,
            )

            _, alarm = adaptive_cusum_detector(
                returns,
                window=WINDOW,
                target_mean=TARGET_MEAN,
                allowance=ALLOWANCE,
                threshold=threshold,
                min_volatility=floor,
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
                f"Average detection delay: "
                f"{np.mean(delays):.2f}"
            )
        else:
            print("Average detection delay: N/A")
