
import numpy as np

from edgeguard.simulator import generate_returns
from edgeguard.detectors import (
    cusum_detector,
    adaptive_cusum_detector,
)

# Experiment 13: Controlled detector comparison

N = 1000
CHANGE_POINT = 500

N_CALIBRATION = 1000
N_TEST = 500

TARGET_MEAN = 0.001
MU_AFTER = -0.001

VOLATILITIES = [0.005, 0.01, 0.02, 0.03]

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001

TARGET_FALSE_ALARM = 0.05

print("EDGEGUARD - Experiment 13")
print("Original vs Adaptive CUSUM")
print("--------------------------------")



# Step 2: Calibrate both detectors

calibrated_thresholds = {}

for sigma in VOLATILITIES:
    original_maxima = []
    adaptive_maxima = []

    for seed in range(N_CALIBRATION):

        returns, _ = generate_returns(
            n=N,
            change_point=N,
            mu_before=TARGET_MEAN,
            mu_after=TARGET_MEAN,
            sigma=sigma,
            seed=seed,
        )

        original_values, _ = cusum_detector(
            returns,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE * sigma,
            threshold=1e100,
        )

        adaptive_values, _ = adaptive_cusum_detector(
            returns,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=1e100,
            min_volatility=MIN_VOLATILITY,
        )

        original_maxima.append(
            float(np.max(original_values))
        )

        adaptive_maxima.append(
            float(np.max(adaptive_values))
        )

    original_threshold = float(
        np.quantile(
            original_maxima,
            1 - TARGET_FALSE_ALARM,
        )
    )

    adaptive_threshold = float(
        np.quantile(
            adaptive_maxima,
            1 - TARGET_FALSE_ALARM,
        )
    )

    calibrated_thresholds[sigma] = {
        "original": original_threshold,
        "adaptive": adaptive_threshold,
    }

    print(f"\nVolatility: {sigma:.3f}")
    print(
        f"Original threshold: "
        f"{original_threshold:.6f}"
    )
    print(
        f"Adaptive threshold: "
        f"{adaptive_threshold:.6f}"
    )



# Step 3: Independent no-change evaluation

print("\nIndependent False-Alarm Comparison")
print("----------------------------------")

for sigma in VOLATILITIES:
    original_false_alarms = 0
    adaptive_false_alarms = 0

    thresholds = calibrated_thresholds[sigma]

    for seed in range(10000, 10000 + N_TEST):
        returns, _ = generate_returns(
            n=N,
            change_point=N,
            mu_before=TARGET_MEAN,
            mu_after=TARGET_MEAN,
            sigma=sigma,
            seed=seed,
        )

        _, original_alarm = cusum_detector(
            returns,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE * sigma,
            threshold=thresholds["original"],
        )

        _, adaptive_alarm = adaptive_cusum_detector(
            returns,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=thresholds["adaptive"],
            min_volatility=MIN_VOLATILITY,
        )

        if original_alarm is not None:
            original_false_alarms += 1

        if adaptive_alarm is not None:
            adaptive_false_alarms += 1

    print(f"\nVolatility: {sigma:.3f}")
    print(
        f"Original CUSUM false alarms: "
        f"{original_false_alarms / N_TEST:.2%}"
    )
    print(
        f"Adaptive CUSUM false alarms: "
        f"{adaptive_false_alarms / N_TEST:.2%}"
    )



# Step 4: Independent regime-change comparison

print("\nIndependent Regime-Change Comparison")
print("------------------------------------")

for sigma in VOLATILITIES:
    thresholds = calibrated_thresholds[sigma]

    results = {
        "Original": {
            "premature": 0,
            "successful": 0,
            "missed": 0,
            "delays": [],
        },
        "Adaptive": {
            "premature": 0,
            "successful": 0,
            "missed": 0,
            "delays": [],
        },
    }

    for seed in range(20000, 20000 + N_TEST):
        returns, _ = generate_returns(
            n=N,
            change_point=CHANGE_POINT,
            mu_before=TARGET_MEAN,
            mu_after=MU_AFTER,
            sigma=sigma,
            seed=seed,
        )

        _, original_alarm = cusum_detector(
            returns,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE * sigma,
            threshold=thresholds["original"],
        )

        _, adaptive_alarm = adaptive_cusum_detector(
            returns,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=thresholds["adaptive"],
            min_volatility=MIN_VOLATILITY,
        )

        for name, alarm in [
            ("Original", original_alarm),
            ("Adaptive", adaptive_alarm),
        ]:
            result = results[name]

            if alarm is None:
                result["missed"] += 1

            elif alarm < CHANGE_POINT:
                result["premature"] += 1

            else:
                result["successful"] += 1
                result["delays"].append(
                    alarm - CHANGE_POINT
                )

    print(f"\nVolatility: {sigma:.3f}")

    for name, result in results.items():
        print(f"\n{name} CUSUM")
        print(
            f"Premature alarms: "
            f"{result['premature'] / N_TEST:.1%}"
        )
        print(
            f"Successful detections: "
            f"{result['successful'] / N_TEST:.1%}"
        )
        print(
            f"Missed detections: "
            f"{result['missed'] / N_TEST:.1%}"
        )

        if result["delays"]:
            print(
                f"Average detection delay: "
                f"{np.mean(result['delays']):.2f}"
            )
        else:
            print("Average detection delay: N/A")
