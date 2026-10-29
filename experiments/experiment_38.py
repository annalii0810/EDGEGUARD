
"""
EDGEGUARD - Experiment 38
Paired Economic Effect Confidence Intervals

Exploratory synthetic research.
No historical test-period data is used.
"""

import numpy as np

from experiment_37 import (
    SEED,
    SIMULATIONS,
    make_returns,
    evaluate_path,
)

BOOTSTRAPS = 3000


def paired_interval(differences, rng):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    estimates = np.empty(BOOTSTRAPS)

    for i in range(BOOTSTRAPS):
        indices = rng.integers(0, n, size=n)
        estimates[i] = differences[indices].mean()

    return np.quantile(estimates, [0.025, 0.975])


def main():
    rng = np.random.default_rng(SEED)
    bootstrap_rng = np.random.default_rng(2038)

    print("EDGEGUARD - Experiment 38")
    print("Paired Economic Effect Confidence Intervals")
    print("=" * 65)

    for scenario in ["No change", "Deterioration"]:
        deteriorated = scenario == "Deterioration"

        return_differences = []
        drawdown_reductions = []

        for _ in range(SIMULATIONS):
            returns = make_returns(rng, deteriorated)
            result = evaluate_path(returns)

            baseline = result["baseline"]
            edgeguard = result["edgeguard"]

            return_differences.append(
                edgeguard["total_return"]
                - baseline["total_return"]
            )

            drawdown_reductions.append(
                baseline["max_drawdown"]
                - edgeguard["max_drawdown"]
            )

        print(f"\n{scenario}")
        print("-" * 65)

        for name, differences in [
            ("Return difference", return_differences),
            ("Drawdown reduction", drawdown_reductions),
        ]:
            values = np.asarray(differences)
            low, high = paired_interval(
                values, bootstrap_rng
            )

            print(f"\n{name}")
            print(f"Mean: {values.mean():+.2%}")
            print(
                f"95% bootstrap CI: "
                f"[{low:+.2%}, {high:+.2%}]"
            )
            print(
                "Positive-effect paths:",
                f"{np.mean(values > 0):.2%}",
            )

    print("\nExploratory paired analysis complete.")
    print("Original Phase 4 findings unchanged.")


if __name__ == "__main__":
    main()
