
import numpy as np

from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control
from edgeguard.transaction_costs import calculate_transaction_costs


# Fixed evaluation settings
N = 1000
N_SIMULATIONS = 100
SEED = 2030

TARGET_MEAN = 0.001

VOLATILITIES = [0.005, 0.010, 0.020, 0.030]
POST_CHANGE_MEANS = [-0.0005, -0.001, -0.002]
CHANGE_POINTS = [300, 500, 700]

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


def evaluate_scenario(sigma, mu_after, change_point, rng):
    return_differences = []
    drawdown_reductions = []
    delays = []

    premature = 0
    successful = 0
    missed = 0

    means = np.full(N, TARGET_MEAN)
    means[change_point:] = mu_after

    for _ in range(N_SIMULATIONS):
        returns = rng.normal(
            loc=means,
            scale=sigma
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
        elif alarm < change_point:
            premature += 1
        else:
            successful += 1
            delays.append(alarm - change_point)

    return {
        "premature": premature / N_SIMULATIONS,
        "successful": successful / N_SIMULATIONS,
        "missed": missed / N_SIMULATIONS,
        "return_gain": float(np.mean(return_differences)),
        "drawdown_reduction": float(np.mean(drawdown_reductions)),
        "average_delay": (
            float(np.mean(delays)) if delays else np.nan
        )
    }


def main():
    rng = np.random.default_rng(SEED)

    print("EDGEGUARD - Experiment 20")
    print("Robustness Across Market Conditions")
    print("=" * 75)

    print(f"Simulations per scenario: {N_SIMULATIONS}")
    print("Post-alarm exposure: 25%")
    print("Transaction costs: 10 bps total")
    print()

    print(
        f"{'Vol':>6} "
        f"{'Mu After':>10} "
        f"{'Change':>7} "
        f"{'Detect':>8} "
        f"{'Premature':>10} "
        f"{'Return Gain':>12} "
        f"{'DD Reduction':>13} "
        f"{'Delay':>9}"
    )

    for sigma in VOLATILITIES:
        for mu_after in POST_CHANGE_MEANS:
            for change_point in CHANGE_POINTS:

                result = evaluate_scenario(
                    sigma,
                    mu_after,
                    change_point,
                    rng
                )

                delay = result["average_delay"]

                delay_text = (
                    f"{delay:.1f}"
                    if np.isfinite(delay)
                    else "N/A"
                )

                print(
                    f"{sigma:>6.3f} "
                    f"{mu_after:>10.4f} "
                    f"{change_point:>7} "
                    f"{result['successful']:>8.1%} "
                    f"{result['premature']:>10.1%} "
                    f"{result['return_gain']:>+12.2%} "
                    f"{result['drawdown_reduction']:>+13.2%} "
                    f"{delay_text:>9}"
                )


if __name__ == "__main__":
    main()
