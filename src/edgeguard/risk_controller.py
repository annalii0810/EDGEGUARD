
import numpy as np


def apply_risk_control(
    returns,
    alarm_index=None,
    reduced_exposure=0.25,
):
    """
    Apply a simulated risk-reduction policy.

    An alarm at index t changes exposure from t+1 onward.
    The alarm observation itself retains full exposure.

    Returns:
        controlled_returns, exposures
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1 or not np.all(np.isfinite(returns)):
        raise ValueError("Returns must be a finite 1D array.")

    if (
        not np.isfinite(reduced_exposure)
        or not 0 <= reduced_exposure <= 1
    ):
        raise ValueError("Reduced exposure must be between 0 and 1.")

    if alarm_index is not None:
        if (
            isinstance(alarm_index, (bool, np.bool_))
            or not isinstance(alarm_index, (int, np.integer))
            or not 0 <= alarm_index < len(returns)
        ):
            raise ValueError("Invalid alarm index.")

    exposures = np.ones(len(returns), dtype=float)

    if alarm_index is not None:
        exposures[alarm_index + 1:] = reduced_exposure

    controlled_returns = returns * exposures

    return controlled_returns, exposures
