
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.historical_detector import historical_adaptive_cusum


DATA_PATH = "data/raw/SPY.csv"

WINDOW = 50
TARGET_MEAN = 0.0002299069244105812
ALLOWANCE = 0.1
THRESHOLD = 25.0
MIN_VOLATILITY = 0.001

SIMULATIONS = 1000
PATH_LENGTH = 250
CHANGE_POINT = 125
BLOCK_LENGTH = 10
DETERIORATION_SHIFT = 0.002

SEED = 92741


def load_training_returns():
    prices = load_historical_data(DATA_PATH)

    strategy = moving_average_strategy(
        prices,
        window=WINDOW,
    )

    train, _, _ = chronological_split(strategy)

    active = train.loc[train["Exposure"] > 0]

    returns = (
        active["StrategyReturn"].to_numpy(dtype=float)
        / active["Exposure"].to_numpy(dtype=float)
    )

    if len(returns) < PATH_LENGTH:
        raise ValueError("Insufficient training data.")

    return returns


def block_bootstrap(rng, returns, length):
    n = len(returns)
    output = np.empty(length, dtype=float)
    position = 0

    while position < length:
        start = rng.integers(0, n - BLOCK_LENGTH + 1)
        block = returns[start:start + BLOCK_LENGTH]

        count = min(BLOCK_LENGTH, length - position)
        output[position:position + count] = block[:count]
        position += count

    return output


def wilson_interval(successes, trials, z=1.95996398454):
    if trials <= 0:
        raise ValueError("Trials must be positive.")

    p = successes / trials
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / trials
            + z2 / (4 * trials * trials)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def detect(returns):
    exposure = np.ones(len(returns))

    _, alarm = historical_adaptive_cusum(
        returns,
        exposure,
        window=WINDOW,
        target_mean=TARGET_MEAN,
        allowance=ALLOWANCE,
        threshold=THRESHOLD,
        min_volatility=MIN_VOLATILITY,
    )

    return alarm


def main():
    print("EDGEGUARD - Experiment 27")
    print("Independent Bootstrap Calibration Evaluation")
    print("=" * 65)

    empirical_returns = load_training_returns()
    rng = np.random.default_rng(SEED)

    false_alarms = 0
    premature = 0
    detected = 0
    missed = 0
    delays = []

    for _ in range(SIMULATIONS):
        baseline = block_bootstrap(
            rng, empirical_returns, PATH_LENGTH
        )

        changed = baseline.copy()
        changed[CHANGE_POINT:] -= DETERIORATION_SHIFT

        null_alarm = detect(baseline)
        changed_alarm = detect(changed)

        if null_alarm is not None:
            false_alarms += 1

        if changed_alarm is None:
            missed += 1
        elif changed_alarm < CHANGE_POINT:
            premature += 1
        else:
            detected += 1
            delays.append(changed_alarm - CHANGE_POINT)

    false_alarm_ci = wilson_interval(
        false_alarms, SIMULATIONS
    )

    detection_ci = wilson_interval(
        detected, SIMULATIONS
    )

    print("\nFrozen Detector Configuration")
    print("-" * 65)
    print("Threshold:", THRESHOLD)
    print("Reference mean:", TARGET_MEAN)
    print("Allowance:", ALLOWANCE)
    print("Minimum volatility:", MIN_VOLATILITY)
    print("Active-return window:", WINDOW)
    print("Monitoring horizon:", PATH_LENGTH)
    print("Bootstrap block length:", BLOCK_LENGTH)
    print("Simulation paths:", SIMULATIONS)
    print("Random seed:", SEED)

    print("\nNo-Change Evaluation")
    print("-" * 65)
    print("False alarms:", false_alarms)
    print(f"False-alarm rate: {false_alarms / SIMULATIONS:.2%}")
    print(
        "95% Wilson CI: "
        f"[{false_alarm_ci[0]:.2%}, {false_alarm_ci[1]:.2%}]"
    )

    print("\nDeterioration Evaluation")
    print("-" * 65)
    print("Premature alarms:", premature)
    print("Successful detections:", detected)
    print("Missed detections:", missed)
    print(f"Premature-alarm rate: {premature / SIMULATIONS:.2%}")
    print(f"Detection rate: {detected / SIMULATIONS:.2%}")
    print(f"Missed rate: {missed / SIMULATIONS:.2%}")
    print(
        "95% Wilson detection CI: "
        f"[{detection_ci[0]:.2%}, {detection_ci[1]:.2%}]"
    )

    if delays:
        print(
            "Average detection delay:",
            f"{np.mean(delays):.2f} trading observations"
        )
        print(
            "Median detection delay:",
            f"{np.median(delays):.2f} trading observations"
        )
    else:
        print("Average detection delay: N/A")

    print("\nResearch Interpretation")
    print("-" * 65)

    if false_alarm_ci[1] <= 0.05:
        print("False-alarm CI upper bound meets the 5% target.")
    else:
        print("False-alarm control is not established at 95% confidence.")

    print(
        "This evaluation uses fresh bootstrap samples "
        "from the same historical training distribution."
    )
    print(
        "It does not establish out-of-sample market performance."
    )


if __name__ == "__main__":
    main()
