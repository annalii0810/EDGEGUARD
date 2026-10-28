
import numpy as np

from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control
from edgeguard.transaction_costs import calculate_transaction_costs


# Experiment settings
N_SIMULATIONS = 500
N_BOOTSTRAP = 2000
N = 1000
CHANGE_POINT = 500

SEED = 2031

TARGET_MEAN = 0.001
MU_AFTER = -0.001
VOLATILITY = 0.01

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001
THRESHOLD = 28.233946

REDUCED_EXPOSURE = 0.25

COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 5.0


def total_return(returns):
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    wealth = np.concatenate((
        [1.0],
        np.cumprod(1.0 + returns)
    ))
    peaks = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peaks))


def bootstrap_ci(values, rng):
    """Percentile bootstrap CI for the mean paired difference."""
    values = np.asarray(values, dtype=float)

    bootstrap_means = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means[i] = np.mean(sample)

    return np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )


def main():
    simulation_rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(SEED + 1)

    return_differences = []
    drawdown_reductions = []

    premature = 0
    successful = 0
    missed = 0

    means = np.full(N, TARGET_MEAN)
    means[CHANGE_POINT:] = MU_AFTER

    for _ in range(N_SIMULATIONS):
        returns = simulation_rng.normal(
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

        costs = calculate_transaction_costs(
            exposures,
            commission_bps=COMMISSION_BPS,
            slippage_bps=SLIPPAGE_BPS,
            initial_exposure=1.0
        )

        net_controlled = controlled - costs

        return_differences.append(
            total_return(net_controlled) - total_return(returns)
        )

        drawdown_reductions.append(
            max_drawdown(returns) - max_drawdown(net_controlled)
        )

        if alarm is None:
            missed += 1
        elif alarm < CHANGE_POINT:
            premature += 1
        else:
            successful += 1

    return_differences = np.asarray(return_differences)
    drawdown_reductions = np.asarray(drawdown_reductions)

    return_ci = bootstrap_ci(
        return_differences,
        bootstrap_rng
    )

    drawdown_ci = bootstrap_ci(
        drawdown_reductions,
        bootstrap_rng
    )

    print("EDGEGUARD - Experiment 21")
    print("Bootstrap Confidence Interval Evaluation")
    print("=" * 55)

    print(f"Simulations: {N_SIMULATIONS}")
    print(f"Bootstrap resamples: {N_BOOTSTRAP}")

    print("\nDetection Performance")
    print(f"Premature alarms: {premature / N_SIMULATIONS:.1%}")
    print(f"Successful detections: {successful / N_SIMULATIONS:.1%}")
    print(f"Missed detections: {missed / N_SIMULATIONS:.1%}")

    print("\nPaired Return Improvement")
    print(
        f"Mean improvement: "
        f"{np.mean(return_differences):+.2%}"
    )
    print(
        f"95% bootstrap CI: "
        f"[{return_ci[0]:+.2%}, {return_ci[1]:+.2%}]"
    )

    print("\nPaired Drawdown Reduction")
    print(
        f"Mean reduction: "
        f"{np.mean(drawdown_reductions):+.2%}"
    )
    print(
        f"95% bootstrap CI: "
        f"[{drawdown_ci[0]:+.2%}, {drawdown_ci[1]:+.2%}]"
    )

    print("\nPaired Outcome Frequencies")
    print(
        f"Return improved: "
        f"{np.mean(return_differences > 0):.1%}"
    )
    print(
        f"Drawdown reduced: "
        f"{np.mean(drawdown_reductions > 0):.1%}"
    )


if __name__ == "__main__":
    main()

