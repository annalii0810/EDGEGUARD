
import numpy as np


def rolling_mean_detector(returns, window=50, threshold=0.0):
    """
    Detect when a rolling average falls below a threshold.

    Returns:
        rolling_means: calculated rolling averages
        first_alarm: index of the first alarm, or None
    """

    # Convert returns into a NumPy array
    returns = np.asarray(returns, dtype=float)

    # Validate the inputs
    if returns.ndim != 1:
        raise ValueError("Returns must be one-dimensional")

    if len(returns) == 0:
        raise ValueError("Returns cannot be empty")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must contain finite numbers")

    if not isinstance(window, int) or not 1 <= window <= len(returns):
        raise ValueError("Invalid window size")

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    # Create an array for rolling averages
    rolling_means = np.full(len(returns), np.nan)

    # Calculate rolling averages
    for t in range(window - 1, len(returns)):
        rolling_means[t] = np.mean(
            returns[t - window + 1:t + 1]
        )

    # Find the first alarm
    first_alarm = None

    for t in range(window - 1, len(returns)):
        if rolling_means[t] < threshold:
            first_alarm = t
            break

    return rolling_means, first_alarm



def cusum_detector(
    returns,
    target_mean=0.001,
    allowance=0.001,
    threshold=0.05,
):
    """
    Downward CUSUM detector for strategy deterioration.

    Returns:
        cusum_values: accumulated evidence at each observation
        first_alarm: first alarm index, or None
    """

    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("Returns must be a nonempty 1D array")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be finite")

    if not np.isfinite(target_mean):
        raise ValueError("Target mean must be finite")

    if not np.isfinite(allowance) or allowance < 0:
        raise ValueError("Allowance must be nonnegative and finite")

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be positive and finite")

    cusum_values = np.zeros(len(returns))
    statistic = 0.0
    first_alarm = None

    for t, observed_return in enumerate(returns):

        statistic = max(
            0.0,
            statistic + target_mean - observed_return - allowance
        )

        cusum_values[t] = statistic

        if first_alarm is None and statistic > threshold:
            first_alarm = t

    return cusum_values, first_alarm



def adaptive_cusum_detector(
    returns,
    window=50,
    target_mean=0.001,
    allowance=0.1,
    threshold=25.0,
    min_volatility=1e-6,
):
    """
    Downward CUSUM using historical volatility estimates.

    At time t, volatility is estimated using only
    returns observed before time t.

    Returns:
        statistics: CUSUM values at each observation
        first_alarm: index of the first alarm, or None
    """
    from edgeguard.volatility import rolling_volatility

    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or len(returns) < 3:
        raise ValueError("Returns must be a 1D array with at least 3 values")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be finite")

    if isinstance(window, (bool, np.bool_)) or not isinstance(
        window, (int, np.integer)
    ):
        raise ValueError("Window must be an integer")

    if window < 2 or window >= len(returns):
        raise ValueError("Invalid window")

    for value in (target_mean, allowance, threshold, min_volatility):
        if not np.isfinite(value):
            raise ValueError("Parameters must be finite")

    if allowance < 0 or threshold <= 0 or min_volatility <= 0:
        raise ValueError("Invalid detector parameters")

    volatility = rolling_volatility(returns, window=window)

    statistics = np.zeros(len(returns))
    statistic = 0.0
    first_alarm = None

    for t in range(window, len(returns)):
        sigma_hat = max(volatility[t], min_volatility)

        normalized_deviation = (
            target_mean - returns[t]
        ) / sigma_hat

        statistic = max(
            0.0,
            statistic + normalized_deviation - allowance
        )

        statistics[t] = statistic

        if first_alarm is None and statistic > threshold:
            first_alarm = t

    return statistics, first_alarm
