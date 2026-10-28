
import numpy as np

from edgeguard.historical_data import load_historical_data
from edgeguard.historical_strategy import moving_average_strategy
from edgeguard.data_split import chronological_split
from edgeguard.historical_detector import historical_adaptive_cusum


DATA_PATH = "data/raw/SPY.csv"

WINDOW = 50
TARGET_MEAN = 0.001
ALLOWANCE = 0.1

THRESHOLDS = [10.0, 15.0, 20.0, 25.0, 35.0, 50.0]
VOLATILITY_FLOORS = [0.0005, 0.001, 0.002]

SIMULATIONS = 200
PATH_LENGTH = 500
CHANGE_POINT = 250

SEED = 2026
TARGET_FALSE_ALARM_RATE = 0.05


def load_training_returns():
    prices = load_historical_data(DATA_PATH)

    strategy = moving_average_strategy(
        prices,
        window=WINDOW,
    )

    train, _, _ = chronological_split(strategy)

    active = train.loc[
        train["Exposure"] > 0
    ].copy()

    # Normalize to per-unit-exposure returns.
    returns = (
        active["StrategyReturn"].to_numpy(dtype=float)
        / active["Exposure"].to_numpy(dtype=float)
    )

    if len(returns) < WINDOW + 2:
        raise ValueError("Insufficient training returns.")

    return returns


def simulate_path(rng, empirical_returns, deteriorate):
    # Independent resampling from training observations.
    sampled = rng.choice(
        empirical_returns,
        size=PATH_LENGTH,
        replace=True,
    )

    if deteriorate:
        # Controlled negative shift after change point.
        sampled[CHANGE_POINT:] -= 0.002

    # All simulated observations are active.
    exposure = np.ones(PATH_LENGTH)

    return sampled, exposure


def evaluate_candidate(
    null_paths,
    changed_paths,
    threshold,
    min_volatility,
):
    false_alarms = 0
    successful_detections = 0
    premature_alarms = 0
    delays = []

    for returns, exposure in null_paths:
        _, alarm = historical_adaptive_cusum(
            returns,
            exposure,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=threshold,
            min_volatility=min_volatility,
        )

        if alarm is not None:
            false_alarms += 1

    for returns, exposure in changed_paths:
        _, alarm = historical_adaptive_cusum(
            returns,
            exposure,
            window=WINDOW,
            target_mean=TARGET_MEAN,
            allowance=ALLOWANCE,
            threshold=threshold,
            min_volatility=min_volatility,
        )

        if alarm is not None and alarm < CHANGE_POINT:
            premature_alarms += 1

        elif alarm is not None:
            successful_detections += 1
            delays.append(alarm - CHANGE_POINT)

    n = len(null_paths)

    return {
        "threshold": threshold,
        "volatility_floor": min_volatility,
        "false_alarm_rate": false_alarms / n,
        "detection_rate": successful_detections / n,
        "premature_rate": premature_alarms / n,
        "average_delay": (
            float(np.mean(delays))
            if delays else float("nan")
        ),
    }


def main():
    print("EDGEGUARD - Experiment 25")
    print("Training-Only CUSUM Calibration")
    print("=" * 75)

    empirical_returns = load_training_returns()

    print("Training active returns:", len(empirical_returns))
    print("Training mean:", round(np.mean(empirical_returns), 6))
    print("Training volatility:", round(np.std(empirical_returns), 6))

    rng = np.random.default_rng(SEED)

    null_paths = [
        simulate_path(rng, empirical_returns, False)
        for _ in range(SIMULATIONS)
    ]

    changed_paths = [
        simulate_path(rng, empirical_returns, True)
        for _ in range(SIMULATIONS)
    ]

    results = []

    print("\nCalibration Results")
    print("-" * 75)
    print(
        f"{'Threshold':>10} {'Vol Floor':>10} "
        f"{'False Alarm':>12} {'Detection':>11} "
        f"{'Premature':>11} {'Avg Delay':>10}"
    )

    for floor in VOLATILITY_FLOORS:
        for threshold in THRESHOLDS:
            result = evaluate_candidate(
                null_paths,
                changed_paths,
                threshold,
                floor,
            )

            results.append(result)

            delay = result["average_delay"]

            delay_text = (
                f"{delay:.1f}"
                if np.isfinite(delay) else "N/A"
            )

            print(
                f"{threshold:>10.1f} "
                f"{floor:>10.4f} "
                f"{result['false_alarm_rate']:>11.1%} "
                f"{result['detection_rate']:>10.1%} "
                f"{result['premature_rate']:>10.1%} "
                f"{delay_text:>10}"
            )

    eligible = [
        r for r in results
        if r["false_alarm_rate"] <= TARGET_FALSE_ALARM_RATE
    ]

    print("\nCalibration Decision")
    print("-" * 75)

    if not eligible:
        print("No candidate meets the false-alarm target.")
        print("Do not select a threshold yet.")
        return

    # Maximize detection, then minimize delay.
    # Prefer lower false alarms as a final tie-breaker.
    selected = sorted(
        eligible,
        key=lambda r: (
            -r["detection_rate"],
            r["average_delay"]
            if np.isfinite(r["average_delay"])
            else float("inf"),
            r["false_alarm_rate"],
        ),
    )[0]

    print("Selected provisional configuration:")
    print("Threshold:", selected["threshold"])
    print("Volatility floor:", selected["volatility_floor"])
    print("False-alarm rate:", f"{selected['false_alarm_rate']:.1%}")
    print("Detection rate:", f"{selected['detection_rate']:.1%}")
    print("Average delay:", selected["average_delay"])

    print("\nSelection is based on training simulations only.")
    print("Independent evaluation is still required.")


if __name__ == "__main__":
    main()
