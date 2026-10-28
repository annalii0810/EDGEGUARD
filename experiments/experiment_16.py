
import numpy as np

from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control


# Experiment settings
N_SIMULATIONS = 500
N = 1000
CHANGE_POINT = 500

TARGET_MEAN = 0.001
MU_AFTER = -0.001
VOLATILITY = 0.01

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001
THRESHOLD = 28.233946

REDUCED_EXPOSURE = 0.25
SEED = 2026


def total_return(returns):
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    wealth = np.concatenate((
        [1.0],
        np.cumprod(1.0 + returns)
    ))
    peaks = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peaks))


def evaluate():
    rng = np.random.default_rng(SEED)

    baseline_returns = []
    controlled_returns = []
    baseline_drawdowns = []
    controlled_drawdowns = []

    premature = 0
    successful = 0
    missed = 0
    detection_delays = []

    means = np.where(
        np.arange(N) < CHANGE_POINT,
        TARGET_MEAN,
        MU_AFTER
    )

    for _ in range(N_SIMULATIONS):
        returns = rng.normal(
            loc=means,
            scale=VOLATILITY
        )

        _, alarm = adaptive_cusum_detector(
            returns,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=THRESHOLD,
            min_volatility=MIN_VOLATILITY
        )

        controlled, exposures = apply_risk_control(
            returns,
            alarm_index=alarm,
            reduced_exposure=REDUCED_EXPOSURE
        )

        baseline_returns.append(total_return(returns))
        controlled_returns.append(total_return(controlled))

        baseline_drawdowns.append(max_drawdown(returns))
        controlled_drawdowns.append(max_drawdown(controlled))

        if alarm is None:
            missed += 1
        elif alarm < CHANGE_POINT:
            premature += 1
        else:
            successful += 1
            detection_delays.append(alarm - CHANGE_POINT)

    baseline_returns = np.array(baseline_returns)
    controlled_returns = np.array(controlled_returns)
    baseline_drawdowns = np.array(baseline_drawdowns)
    controlled_drawdowns = np.array(controlled_drawdowns)

    print("EDGEGUARD - Experiment 16")
    print("Monte Carlo Risk Controller Evaluation")
    print("--------------------------------------")

    print(f"Simulations: {N_SIMULATIONS}")

    print("\nDetection Performance")
    print(f"Premature alarms: {premature / N_SIMULATIONS:.1%}")
    print(f"Successful detections: {successful / N_SIMULATIONS:.1%}")
    print(f"Missed detections: {missed / N_SIMULATIONS:.1%}")

    if detection_delays:
        print(
            f"Average detection delay: "
            f"{np.mean(detection_delays):.2f}"
        )

    print("\nAverage Trading Performance")
    print(
        f"Baseline average total return: "
        f"{np.mean(baseline_returns):.2%}"
    )
    print(
        f"Controlled average total return: "
        f"{np.mean(controlled_returns):.2%}"
    )
    print(
        f"Baseline average max drawdown: "
        f"{np.mean(baseline_drawdowns):.2%}"
    )
    print(
        f"Controlled average max drawdown: "
        f"{np.mean(controlled_drawdowns):.2%}"
    )

    print("\nPaired Comparison")
    print(
        f"Controlled strategy improved return: "
        f"{np.mean(controlled_returns > baseline_returns):.1%}"
    )
    print(
        f"Controlled strategy reduced drawdown: "
        f"{np.mean(controlled_drawdowns < baseline_drawdowns):.1%}"
    )

    print(
        f"Average return difference: "
        f"{np.mean(controlled_returns - baseline_returns):+.2%}"
    )
    print(
        f"Average drawdown reduction: "
        f"{np.mean(baseline_drawdowns - controlled_drawdowns):+.2%}"
    )


if __name__ == "__main__":
    evaluate()
