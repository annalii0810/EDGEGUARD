
import numpy as np


def rolling_volatility(returns, window=50):
    """
    Estimate sample volatility using past returns only.

    At index t, use observations [t-window, ..., t-1].

    Returns:
        NumPy array containing volatility estimates.
        The first `window` entries are NaN.
    """

    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or len(returns) == 0:
        raise ValueError("Returns must be a nonempty 1D array")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be finite")

    if isinstance(window, (bool, np.bool_)) or not isinstance(
        window, (int, np.integer)
    ):
        raise ValueError("Window must be an integer")

    if window < 2 or window >= len(returns):
        raise ValueError("Invalid volatility window")

    volatility = np.full(len(returns), np.nan)

    for t in range(window, len(returns)):
        past_returns = returns[t - window:t]

        volatility[t] = np.std(
            past_returns,
            ddof=1
        )

    return volatility
