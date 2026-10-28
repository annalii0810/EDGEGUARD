
import numpy as np

from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control


# Experiment settings
N_SIMULATIONS = 500
N = 1000
SEED = 2027

TARGET_MEAN = 0.001
VOLATILITY = 0.01

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001
THRESHOLD = 28.233946

REDUCED_EXPOSURE = 0.25


def total_return(returns):
    """Calculate compounded total return."""
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    """Calculate maximum peak-to-trough drawdown."""
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

    false_alarms = 0
    alarm_indices = []

    for _ in range(N_SIMULATIONS):

        # No regime change: expected return remains positive.
        returns = rng.normal(
            loc=TARGET_MEAN,
            scale=VOLATILITY,
            size=N
        )

        # Detect possible deterioration.
        _, alarm = adaptive_cusum_detector(
            returns,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=THRESHOLD,
            min_volatility=MIN_VOLATILITY
        )

        # Apply exposure reduction after the alarm.
        controlled, exposures = apply_risk_control(
            returns,
            alarm_index=alarm,
            reduced_exposure=REDUCED_EXPOSURE
        )

        baseline_returns.append(total_return(returns))
        controlled_returns.append(total_return(controlled))

        baseline_drawdowns.append(max_drawdown(returns))
        controlled_drawdowns.append(max_drawdown(controlled))

        if alarm is not None:
            false_alarms += 1
            alarm_indices.append(alarm)

    baseline_returns = np.array(baseline_returns)
    controlled_returns = np.array(controlled_returns)

    baseline_drawdowns = np.array(baseline_drawdowns)
    controlled_drawdowns = np.array(controlled_drawdowns)

    return_differences = controlled_returns - baseline_returns
    drawdown_differences = baseline_drawdowns - controlled_drawdowns

    print("EDGEGUARD - Experiment 17")
    print("False-Alarm Economic Cost Evaluation")
    print("------------------------------------")

    print(f"Simulations: {N_SIMULATIONS}")
    print("Regime change: None")

    print("\nFalse-Alarm Performance")
    print(f"False alarms: {false_alarms}")
    print(
        f"False-alarm proportion: "
        f"{false_alarms / N_SIMULATIONS:.2%}"
    )

    if alarm_indices:
        print(
            f"Average false-alarm index: "
            f"{np.mean(alarm_indices):.2f}"
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
        f"Average return difference: "
        f"{np.mean(return_differences):+.2%}"
    )
    print(
        f"Average drawdown reduction: "
        f"{np.mean(drawdown_differences):+.2%}"
    )

    print(
        f"Controlled strategy improved return: "
        f"{np.mean(return_differences > 0):.1%}"
    )

    print(
        f"Controlled strategy reduced drawdown: "
        f"{np.mean(drawdown_differences > 0):.1%}"
    )

    if false_alarms:
        print("\nFalse-Alarm Conditional Cost")

        print(
            f"Average return difference after false alarms: "
            f"{np.mean(return_differences[
                np.abs(return_differences) > 1e-12
            ]):+.2%}"
        )


if __name__ == "__main__":
    evaluate()
