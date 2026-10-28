
import numpy as np


def historical_adaptive_cusum(
    returns,
    exposure,
    window=50,
    target_mean=0.001,
    allowance=0.1,
    threshold=25.0,
    min_volatility=0.001,
):
    """
    Exposure-aware downward CUSUM detector.

    Only active strategy returns contribute to the
    volatility estimate and CUSUM statistic.

    An alarm at t can affect exposure from t+1.
    """

    returns = np.asarray(returns, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    if returns.ndim != 1 or exposure.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if len(returns) != len(exposure) or len(returns) < 3:
        raise ValueError("Invalid input lengths.")

    if not np.isfinite(returns).all():
        raise ValueError("Returns must be finite.")

    if not np.isfinite(exposure).all():
        raise ValueError("Exposure must be finite.")

    if np.any((exposure < 0) | (exposure > 1)):
        raise ValueError("Exposure must be between 0 and 1.")

    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 2
    ):
        raise ValueError("Invalid window.")

    for value in (
        target_mean, allowance, threshold, min_volatility
    ):
        if not np.isfinite(value):
            raise ValueError("Parameters must be finite.")

    if allowance < 0 or threshold <= 0 or min_volatility <= 0:
        raise ValueError("Invalid detector parameters.")

    statistics = np.zeros(len(returns))
    first_alarm = None
    statistic = 0.0

    # Store historical active returns only.
    active_history = []

    for t in range(len(returns)):
        if exposure[t] > 0:
            # Convert realized strategy returns into
            # returns per unit of active exposure.
            active_return = returns[t] / exposure[t]

            if len(active_history) >= window:
                past = np.asarray(
                    active_history[-window:]
                )

                sigma = max(
                    float(np.std(past, ddof=1)),
                    min_volatility,
                )

                deviation = (
                    target_mean - active_return
                ) / sigma

                statistic = max(
                    0.0,
                    statistic + deviation - allowance
                )

                if (
                    first_alarm is None
                    and statistic > threshold
                ):
                    first_alarm = t

            active_history.append(active_return)

        # Inactive periods leave CUSUM unchanged.
        statistics[t] = statistic

    return statistics, first_alarm
