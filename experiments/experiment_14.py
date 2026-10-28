
import numpy as np

from edgeguard.risk_controller import apply_risk_control


def max_drawdown(returns):
    """Maximum peak-to-trough loss using compounded returns."""
    wealth = np.concatenate((
        [1.0],
        np.cumprod(1.0 + returns)
    ))

    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / peaks

    return float(np.max(drawdowns))


def total_return(returns):
    """Compounded cumulative return."""
    return float(np.prod(1.0 + returns) - 1.0)


# Reproducible synthetic experiment
rng = np.random.default_rng(42)

N = 1000
CHANGE_POINT = 500
ALARM_INDEX = 550

# Positive expected returns before deterioration,
# negative expected returns afterward.
means = np.where(
    np.arange(N) < CHANGE_POINT,
    0.001,
    -0.001
)

returns = rng.normal(
    loc=means,
    scale=0.01
)

# Baseline: always fully invested
baseline_returns = returns.copy()

# Controlled: reduce exposure after the alarm
controlled_returns, exposures = apply_risk_control(
    returns,
    alarm_index=ALARM_INDEX,
    reduced_exposure=0.25
)

print("EDGEGUARD - Experiment 14")
print("Risk Controller Baseline Evaluation")
print("-----------------------------------")

print(f"Change point: {CHANGE_POINT}")
print(f"Alarm index: {ALARM_INDEX}")
print(f"Exposure after alarm: {exposures[-1]:.0%}")

print("\nUncontrolled Strategy")
print(f"Total return: {total_return(baseline_returns):.2%}")
print(f"Maximum drawdown: {max_drawdown(baseline_returns):.2%}")

print("\nControlled Strategy")
print(f"Total return: {total_return(controlled_returns):.2%}")
print(f"Maximum drawdown: {max_drawdown(controlled_returns):.2%}")
