
import numpy as np


def rolling_mean_detector(
    returns,
    window=50,
    target_mean=0.00023,
    threshold=2.0,
    reference_volatility=0.00715,
):
    """
    Detect negative mean shifts using a rolling
    standardized mean statistic.

    At time t, the window includes return[t].
    An alarm can affect trading only from t+1.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("Returns must be finite and 1D.")

    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 2
        or window > len(returns)
    ):
        raise ValueError("Invalid window.")

    if (
        not np.isfinite(target_mean)
        or not np.isfinite(threshold)
        or not np.isfinite(reference_volatility)
        or threshold <= 0
        or reference_volatility <= 0
    ):
        raise ValueError("Invalid parameters.")

    statistics = np.zeros(len(returns))
    first_alarm = None

    for t in range(window - 1, len(returns)):
        recent = returns[t - window + 1:t + 1]

        statistic = (
            (target_mean - np.mean(recent))
            * np.sqrt(window)
            / reference_volatility
        )

        statistics[t] = max(0.0, statistic)

        if first_alarm is None and statistic > threshold:
            first_alarm = t

    return statistics, first_alarm


def standardized_cusum_detector(
    returns,
    target_mean=0.00023,
    reference_volatility=0.00715,
    allowance=0.1,
    threshold=25.0,
    warmup=50,
):
    """
    One-sided downward CUSUM with fixed
    reference volatility.

    The first warmup observations do not
    contribute to the statistic.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or not np.isfinite(returns).all():
        raise ValueError("Returns must be finite and 1D.")

    if (
        isinstance(warmup, (bool, np.bool_))
        or not isinstance(warmup, (int, np.integer))
        or warmup < 0
        or warmup >= len(returns)
    ):
        raise ValueError("Invalid warmup.")

    if (
        not np.isfinite(target_mean)
        or not np.isfinite(reference_volatility)
        or not np.isfinite(allowance)
        or not np.isfinite(threshold)
        or reference_volatility <= 0
        or allowance < 0
        or threshold <= 0
    ):
        raise ValueError("Invalid parameters.")

    statistics = np.zeros(len(returns))
    statistic = 0.0
    first_alarm = None

    for t in range(warmup, len(returns)):
        deviation = (
            target_mean - returns[t]
        ) / reference_volatility

        statistic = max(
            0.0,
            statistic + deviation - allowance
        )

        statistics[t] = statistic

        if first_alarm is None and statistic > threshold:
            first_alarm = t

    return statistics, first_alarm
