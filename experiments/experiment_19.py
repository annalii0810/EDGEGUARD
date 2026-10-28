
import numpy as np

from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control
from edgeguard.transaction_costs import calculate_transaction_costs


N_SIMULATIONS = 500
N = 1000
CHANGE_POINT = 500
SEED = 2029

TARGET_MEAN = 0.001
MU_AFTER = -0.001
VOLATILITY = 0.01

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001
THRESHOLD = 28.233946

EXPOSURES = [1.0, 0.75, 0.50, 0.25, 0.0]

COST_SCENARIOS = {
    "Low": (1.0, 1.0),
    "Medium": (5.0, 5.0),
    "High": (15.0, 15.0),
}


def total_return(returns):
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    wealth = np.concatenate((
        [1.0],
        np.cumprod(1.0 + returns)
    ))
    peaks = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / peaks))


def run_scenario(regime_change, seed):
    rng = np.random.default_rng(seed)

    means = np.full(N, TARGET_MEAN)

    if regime_change:
        means[CHANGE_POINT:] = MU_AFTER

    results = {
        name: {
            exposure: {"returns": [], "drawdowns": []}
            for exposure in EXPOSURES
        }
        for name in COST_SCENARIOS
    }

    alarm_count = 0

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

        if alarm is not None:
            alarm_count += 1

        for exposure in EXPOSURES:
            gross_returns, weights = apply_risk_control(
                returns,
                alarm_index=alarm,
                reduced_exposure=exposure
            )

            for name, (commission, slippage) in COST_SCENARIOS.items():
                costs = calculate_transaction_costs(
                    weights,
                    commission_bps=commission,
                    slippage_bps=slippage,
                    initial_exposure=1.0
                )

                # Simplified end-of-period cost convention.
                # All policies begin fully invested.
                # Costs are deducted when exposure changes.
                net_returns = gross_returns - costs

                results[name][exposure]["returns"].append(
                    total_return(net_returns)
                )
                results[name][exposure]["drawdowns"].append(
                    max_drawdown(net_returns)
                )

    return results, alarm_count


def print_results(title, results, alarm_count):
    print(f"\n{title}")
    print("=" * 65)
    print(f"Paths with an alarm: {alarm_count / N_SIMULATIONS:.1%}")

    for name, (commission, slippage) in COST_SCENARIOS.items():
        print(
            f"\n{name} costs "
            f"({commission + slippage:.0f} bps total)"
        )

        print(
            f"{'Exposure':<12}"
            f"{'Avg Return':>15}"
            f"{'Avg Drawdown':>18}"
            f"{'Return Improved':>20}"
        )

        baseline = np.asarray(results[name][1.0]["returns"])

        for exposure in EXPOSURES:
            values = results[name][exposure]
            returns = np.asarray(values["returns"])
            drawdowns = np.asarray(values["drawdowns"])

            print(
                f"{exposure:<12.0%}"
                f"{np.mean(returns):>15.2%}"
                f"{np.mean(drawdowns):>18.2%}"
                f"{np.mean(returns > baseline):>20.1%}"
            )


def main():
    print("EDGEGUARD - Experiment 19")
    print("Transaction Cost Sensitivity")
    print("=" * 40)
    print(f"Simulations per scenario: {N_SIMULATIONS}")

    change_results, change_alarms = run_scenario(
        regime_change=True,
        seed=SEED
    )

    print_results(
        "Scenario A: Strategy Deterioration",
        change_results,
        change_alarms
    )

    stable_results, stable_alarms = run_scenario(
        regime_change=False,
        seed=SEED + 1
    )

    print_results(
        "Scenario B: No Regime Change",
        stable_results,
        stable_alarms
    )


if __name__ == "__main__":
    main()
