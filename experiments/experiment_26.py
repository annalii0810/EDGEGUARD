
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.historical_detector import historical_adaptive_cusum


DATA_PATH = "data/raw/SPY.csv"

WINDOW = 50
ALLOWANCE = 0.1
MIN_VOLATILITY = 0.001

THRESHOLDS = [10, 15, 20, 25, 35, 50, 75, 100, 150]

SIMULATIONS = 300
PATH_LENGTH = 250
CHANGE_POINT = 125
BLOCK_LENGTH = 10

DETERIORATION_SHIFT = 0.002
TARGET_FALSE_ALARM_RATE = 0.05

SEED = 2026


def load_training_returns():
    prices = load_historical_data(DATA_PATH)

    strategy = moving_average_strategy(
        prices,
        window=WINDOW,
    )

    train, _, _ = chronological_split(strategy)

    active = train.loc[
        train["Exposure"] > 0
    ]

    returns = (
        active["StrategyReturn"].to_numpy(dtype=float)
        / active["Exposure"].to_numpy(dtype=float)
    )

    if len(returns) < PATH_LENGTH:
        raise ValueError("Insufficient training observations.")

    return returns


def block_bootstrap(rng, returns, length):
    n = len(returns)
    blocks = []

    while sum(len(block) for block in blocks) < length:
        start = rng.integers(
            0, n - BLOCK_LENGTH + 1
        )

        blocks.append(
            returns[start:start + BLOCK_LENGTH]
        )

    return np.concatenate(blocks)[:length]


def generate_scenarios(rng, empirical_returns):
    null_paths = []
    changed_paths = []

    for _ in range(SIMULATIONS):
        path = block_bootstrap(
            rng, empirical_returns, PATH_LENGTH
        )

        null_paths.append(path)

        changed = path.copy()
        changed[CHANGE_POINT:] -= DETERIORATION_SHIFT
        changed_paths.append(changed)

    return null_paths, changed_paths


def evaluate_candidate(
    null_paths,
    changed_paths,
    target_mean,
    threshold,
):
    false_alarms = 0
    premature = 0
    detected = 0
    delays = []

    exposure = np.ones(PATH_LENGTH)

    for null_path, changed_path in zip(
        null_paths, changed_paths
    ):
        _, null_alarm = historical_adaptive_cusum(
            null_path,
            exposure,
            window=WINDOW,
            target_mean=target_mean,
            allowance=ALLOWANCE,
            threshold=threshold,
            min_volatility=MIN_VOLATILITY,
        )

        if null_alarm is not None:
            false_alarms += 1

        _, changed_alarm = historical_adaptive_cusum(
            changed_path,
            exposure,
            window=WINDOW,
            target_mean=target_mean,
            allowance=ALLOWANCE,
            threshold=threshold,
            min_volatility=MIN_VOLATILITY,
        )

        if changed_alarm is not None:
            if changed_alarm < CHANGE_POINT:
                premature += 1
            else:
                detected += 1
                delays.append(
                    changed_alarm - CHANGE_POINT
                )

    n = len(null_paths)

    return {
        "threshold": threshold,
        "false_alarm": false_alarms / n,
        "premature": premature / n,
        "detection": detected / n,
        "delay": (
            float(np.mean(delays))
            if delays else float("nan")
        ),
    }


def main():
    print("EDGEGUARD - Experiment 26")
    print("Training-Only Block Bootstrap Calibration")
    print("=" * 70)

    empirical_returns = load_training_returns()

    target_mean = float(np.mean(empirical_returns))

    print("Training observations:", len(empirical_returns))
    print(f"Training reference mean: {target_mean:.8f}")
    print(f"Training volatility: {np.std(empirical_returns, ddof=1):.6f}")
    print("Monitoring horizon:", PATH_LENGTH)
    print("Block length:", BLOCK_LENGTH)
    print("Simulations:", SIMULATIONS)

    rng = np.random.default_rng(SEED)

    null_paths, changed_paths = generate_scenarios(
        rng, empirical_returns
    )

    results = []

    print("\nCalibration Results")
    print("-" * 70)
    print(
        f"{'Threshold':>10} "
        f"{'False Alarm':>12} "
        f"{'Premature':>12} "
        f"{'Detection':>12} "
        f"{'Avg Delay':>12}"
    )

    for threshold in THRESHOLDS:
        result = evaluate_candidate(
            null_paths,
            changed_paths,
            target_mean,
            threshold,
        )

        results.append(result)

        delay = result["delay"]

        delay_text = (
            f"{delay:.1f}"
            if np.isfinite(delay)
            else "N/A"
        )

        print(
            f"{threshold:>10.1f} "
            f"{result['false_alarm']:>11.1%} "
            f"{result['premature']:>11.1%} "
            f"{result['detection']:>11.1%} "
            f"{delay_text:>12}"
        )

    eligible = [
        result for result in results
        if result["false_alarm"] <= TARGET_FALSE_ALARM_RATE
    ]

    print("\nCalibration Decision")
    print("-" * 70)

    if not eligible:
        print("No candidate meets the false-alarm target.")
        print("Do not select a threshold yet.")
        return

    selected = sorted(
        eligible,
        key=lambda r: (
            -r["detection"],
            r["delay"] if np.isfinite(r["delay"]) else float("inf"),
            r["false_alarm"],
        ),
    )[0]

    print("Selected provisional threshold:", selected["threshold"])
    print("Training reference mean:", target_mean)
    print("False-alarm rate:", f"{selected['false_alarm']:.1%}")
    print("Premature-alarm rate:", f"{selected['premature']:.1%}")
    print("Detection rate:", f"{selected['detection']:.1%}")
    print("Average detection delay:", selected["delay"])

    print("\nThis configuration requires independent evaluation.")


if __name__ == "__main__":
    main()
