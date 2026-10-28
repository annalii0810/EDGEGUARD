
import numpy as np
import pandas as pd


def run_backtest(
    data,
    exposure_column="Exposure",
    return_column="Return",
    transaction_cost_bps=5.0,
    initial_exposure=0.0,
):
    """
    Backtest a precomputed, causal exposure strategy.

    Exposure[t] applies to Return[t].
    Transaction costs are charged when exposure changes.

    Returns:
        results DataFrame, performance metrics dictionary
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a DataFrame.")

    if data.empty:
        raise ValueError("Data cannot be empty.")

    if not {exposure_column, return_column}.issubset(data.columns):
        raise ValueError("Missing exposure or return column.")

    if (
        isinstance(transaction_cost_bps, bool)
        or not isinstance(
            transaction_cost_bps, (int, float, np.number)
        )
        or not np.isfinite(transaction_cost_bps)
        or transaction_cost_bps < 0
    ):
        raise ValueError("Invalid transaction costs.")

    if (
        isinstance(initial_exposure, bool)
        or not isinstance(initial_exposure, (int, float, np.number))
        or not np.isfinite(initial_exposure)
        or not 0 <= initial_exposure <= 1
    ):
        raise ValueError("Invalid initial exposure.")

    result = data.copy()

    exposure = pd.to_numeric(
        result[exposure_column], errors="raise"
    ).to_numpy(dtype=float)

    returns = pd.to_numeric(
        result[return_column], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(exposure).all():
        raise ValueError("Exposure must be finite.")

    if not np.isfinite(returns).all():
        raise ValueError("Returns must be finite.")

    if np.any((exposure < 0) | (exposure > 1)):
        raise ValueError("Exposure must be between 0 and 1.")

    if np.any(returns <= -1):
        raise ValueError("Returns must be greater than -100%.")

    previous_exposure = np.concatenate(
        ([float(initial_exposure)], exposure[:-1])
    )

    turnover = np.abs(exposure - previous_exposure)
    costs = turnover * (transaction_cost_bps / 10000.0)

    gross_returns = exposure * returns
    net_returns = gross_returns - costs

    if np.any(net_returns <= -1):
        raise ValueError("Net returns cannot be <= -100%.")

    equity = np.cumprod(1.0 + net_returns)

    # Include starting equity of 1 when measuring drawdown.
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], equity))
    )[1:]

    drawdown = equity / running_peak - 1.0

    result["Turnover"] = turnover
    result["TransactionCost"] = costs
    result["GrossReturn"] = gross_returns
    result["NetReturn"] = net_returns
    result["Equity"] = equity
    result["Drawdown"] = drawdown

    total_return = float(equity[-1] - 1.0)
    max_drawdown = float(-np.min(drawdown))

    std = float(np.std(net_returns, ddof=1)) if len(net_returns) > 1 else 0.0

    sharpe = (
        float(np.sqrt(252) * np.mean(net_returns) / std)
        if std > 0
        else np.nan
    )

    metrics = {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "annualized_sharpe": sharpe,
        "total_transaction_cost": float(np.sum(costs)),
        "total_turnover": float(np.sum(turnover)),
        "observations": len(result),
    }

    return result, metrics
