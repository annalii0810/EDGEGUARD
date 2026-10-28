
import numpy as np


def calculate_transaction_costs(
    exposures,
    commission_bps=5.0,
    slippage_bps=5.0,
    initial_exposure=1.0,
):
    """
    Calculate transaction costs from changes in exposure.

    Parameters
    ----------
    exposures : array-like
        Exposure applied to each period's return.
    commission_bps : float
        Commission in basis points.
    slippage_bps : float
        Slippage in basis points.
    initial_exposure : float
        Exposure already held before the first period.

    Returns
    -------
    numpy.ndarray
        Transaction cost for each period, expressed
        as a fraction of portfolio value.
    """
    exposures = np.asarray(exposures, dtype=float)

    if exposures.ndim != 1:
        raise ValueError("Exposures must be one-dimensional.")

    if not np.all(np.isfinite(exposures)):
        raise ValueError("Exposures must be finite.")

    if np.any((exposures < 0) | (exposures > 1)):
        raise ValueError("Exposures must be between 0 and 1.")

    for value in (
        commission_bps,
        slippage_bps,
    ):
        if not np.isscalar(value):
            raise ValueError("Costs must be scalar values.")

        if not np.isfinite(value) or value < 0:
            raise ValueError("Costs must be nonnegative and finite.")

    if (
        not np.isscalar(initial_exposure)
        or not np.isfinite(initial_exposure)
        or not 0 <= initial_exposure <= 1
    ):
        raise ValueError("Invalid initial exposure.")

    previous = np.concatenate((
        [initial_exposure],
        exposures[:-1]
    ))

    turnover = np.abs(exposures - previous)

    cost_rate = (
        commission_bps + slippage_bps
    ) / 10000.0

    return turnover * cost_rate
