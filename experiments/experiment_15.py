
import numpy as np

from edgeguard.detectors import adaptive_cusum_detector
from edgeguard.risk_controller import apply_risk_control


def total_return(returns):
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    wealth = np.concatenate((
        [1.0],
        np.cumprod(1.0 + returns)
    ))

    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / peaks

    return float(np.max(drawdowns))


# Experiment settings
N = 1000
CHANGE_POINT = 500
SEED = 42

TARGET_MEAN = 0.001
MU_AFTER = -0.001
VOLATILITY = 0.01

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001

# Previously calibrated adaptive CUSUM threshold
THRESHOLD = 28.233946

# Generate reproducible synthetic returns
rng = np.random.default_rng(SEED)

means = np.where(
    np.arange(N) < CHANGE_POINT,
    TARGET_MEAN,
    MU_AFTER
)

returns = rng.normal(
    loc=means,
    scale=VOLATILITY
)

# Detect deterioration using observed returns
statistics, alarm_index = adaptive_cusum_detector(
    returns,
    window=WINDOW,
    target_mean=TARGET_MEAN,
    allowance=ALLOWANCE,
    threshold=THRESHOLD,
    min_volatility=MIN_VOLATILITY
)

# Apply risk control
controlled_returns, exposures = apply_risk_control(
    returns,
    alarm_index=alarm_index,
    reduced_exposure=0.25
)

# Report results
print("EDGEGUARD - Experiment 15")
print("Adaptive CUSUM Risk Controller")
print("--------------------------------")

print(f"Change point: {CHANGE_POINT}")

if alarm_index is None:
    print("Alarm: None")
    print("Detection status: Missed detection")
elif alarm_index < CHANGE_POINT:
    print(f"Alarm index: {alarm_index}")
    print("Detection status: Premature alarm")
else:
    print(f"Alarm index: {alarm_index}")
    print("Detection status: Successful detection")
    print(f"Detection delay: {alarm_index - CHANGE_POINT}")

print("\nUncontrolled Strategy")
print(f"Total return: {total_return(returns):.2%}")
print(f"Maximum drawdown: {max_drawdown(returns):.2%}")

print("\nControlled Strategy")
print(f"Total return: {total_return(controlled_returns):.2%}")
print(f"Maximum drawdown: {max_drawdown(controlled_returns):.2%}")

print(f"\nFinal exposure: {exposures[-1]:.0%}")
