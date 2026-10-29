
import math


def wilson_interval(successes, trials, z=1.959963984540054):
    """Calculate a 95% Wilson confidence interval."""

    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials <= 0
        or not 0 <= successes <= trials
    ):
        raise ValueError("Invalid successes or trials.")

    if not math.isfinite(z) or z <= 0:
        raise ValueError("Invalid z-score.")

    p = successes / trials
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator

    margin = (
        z * math.sqrt(
            p * (1 - p) / trials
            + z2 / (4 * trials * trials)
        ) / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )
