
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

EXPOSURE_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.0]
SEED = 2028


def total_return(returns):
    """Compounded cumulative return."""
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown."""
    wealth = np.concatenate((
        [1.0],
        np.cumprod(1.0 + returns)
    ))

    peaks = np.maximum.accumulate(wealth)

    return float(np.max(1.0 - wealth / peaks))


def run_scenario(regime_change, seed):
    rng = np.random.default_rng(seed)

    results = {
        exposure: {"returns": [], "drawdowns": []}
        for exposure in EXPOSURE_LEVELS
    }

    premature = 0
    successful = 0
    missed = 0
    false_alarms = 0

    means = np.full(N, TARGET_MEAN)

    if regime_change:
        means[CHANGE_POINT:] = MU_AFTER

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

        if regime_change:
            if alarm is None:
                missed += 1
            elif alarm < CHANGE_POINT:
                premature += 1
            else:
                successful += 1
        elif alarm is not None:
            false_alarms += 1

        for exposure in EXPOSURE_LEVELS:
            controlled, _ = apply_risk_control(
                returns,
                alarm_index=alarm,
                reduced_exposure=exposure
            )

            results[exposure]["returns"].append(
                total_return(controlled)
            )

            results[exposure]["drawdowns"].append(
                max_drawdown(controlled)
            )

    return results, {
        "premature": premature,
        "successful": successful,
        "missed": missed,
        "false_alarms": false_alarms
    }


def print_results(title, results, counts, regime_change):
    print(f"\n{title}")
    print("-" * 65)

    if regime_change:
        print(
            f"Premature alarms: "
            f"{counts['premature'] / N_SIMULATIONS:.1%}"
        )
        print(
            f"Successful detections: "
            f"{counts['successful'] / N_SIMULATIONS:.1%}"
        )
        print(
            f"Missed detections: "
            f"{counts['missed'] / N_SIMULATIONS:.1%}"
        )
    else:
        print(
            f"False alarms: "
            f"{counts['false_alarms'] / N_SIMULATIONS:.1%}"
        )

    print(
        f"\n{'Exposure':<12}"
        f"{'Avg Return':>15}"
        f"{'Avg Drawdown':>18}"
        f"{'Return Improved':>20}"
    )

    baseline = np.array(results[1.0]["returns"])

    for exposure in EXPOSURE_LEVELS:
        returns = np.array(results[exposure]["returns"])
        drawdowns = np.array(results[exposure]["drawdowns"])

        improved = np.mean(returns > baseline)

        print(
            f"{exposure:>8.0%}    "
            f"{np.mean(returns):>14.2%}"
            f"{np.mean(drawdowns):>18.2%}"
            f"{improved:>20.1%}"
        )


def main():
    print("EDGEGUARD - Experiment 18")
    print("Risk-Reduction Policy Comparison")
    print("================================")

    print(f"Simulations per scenario: {N_SIMULATIONS}")
    print(f"Exposure policies: {EXPOSURE_LEVELS}")

    change_results, change_counts = run_scenario(
        regime_change=True,
        seed=SEED
    )

    print_results(
        "Scenario A: Strategy Deterioration",
        change_results,
        change_counts,
        regime_change=True
    )

    stable_results, stable_counts = run_scenario(
        regime_change=False,
        seed=SEED + 1
    )

    print_results(
        "Scenario B: No Regime Change",
        stable_results,
        stable_counts,
        regime_change=False
    )


if __name__ == "__main__":
    main()
