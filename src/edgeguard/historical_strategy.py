
import numpy as np
import pandas as pd


def moving_average_strategy(data, window=50):
    """
    Causal moving-average trading strategy.

    Signal calculated at the previous close
    determines exposure for the next return.

    Requires:
        Date, Close, Return
    """

    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window < 2
    ):
        raise ValueError("Window must be an integer >= 2.")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a DataFrame.")

    required = {"Date", "Close", "Return"}

    if not required.issubset(data.columns):
        raise ValueError("Missing required columns.")

    result = data.copy()

    if result.empty:
        raise ValueError("Data cannot be empty.")

    dates = pd.to_datetime(result["Date"], errors="raise")

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be valid and unique.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Data must be chronologically sorted.")

    for column in ["Close", "Return"]:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise"
        )

        if not np.isfinite(result[column]).all():
            raise ValueError("Prices and returns must be finite.")

    if (result["Close"] <= 0).any():
        raise ValueError("Closing prices must be positive.")

    
    # Verify that supplied returns match closing prices.
    expected_returns = result["Close"].pct_change()

    if len(result) > 1:
        if not np.allclose(
            result["Return"].iloc[1:].to_numpy(),
            expected_returns.iloc[1:].to_numpy(),
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(
                "Return values do not match closing prices."
            )


    result["Date"] = dates

    result["MovingAverage"] = (
        result["Close"].rolling(window=window).mean()
    )

    # Yesterday's signal determines today's position.
    signal = (
        result["Close"] > result["MovingAverage"]
    )

    result["Exposure"] = (
        signal.shift(1).fillna(False).astype(float)
    )

    result["StrategyReturn"] = (
        result["Exposure"] * result["Return"]
    )

    return result
