
import numpy as np
import pandas as pd


def load_historical_data(file_path):
    """
    Load and validate historical market data.

    Required CSV columns:
        Date, Close

    Returns a DataFrame containing:
        Date, Close, Return
    """

    data = pd.read_csv(file_path)

    required_columns = {"Date", "Close"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "CSV must contain Date and Close columns."
        )

    data = data[["Date", "Close"]].copy()

    # Validate dates
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise"
    )

    # Validate closing prices
    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="raise"
    )

    if data.empty:
        raise ValueError("Historical data cannot be empty.")

    if data["Date"].isna().any():
        raise ValueError("Dates cannot be missing.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates are not allowed.")

    if not np.isfinite(data["Close"]).all():
        raise ValueError("Closing prices must be finite.")

    if (data["Close"] <= 0).any():
        raise ValueError("Closing prices must be positive.")

    # Sort chronologically
    data = data.sort_values("Date").reset_index(drop=True)

    # Calculate simple daily returns
    data["Return"] = data["Close"].pct_change()

    # Remove first row without a previous price
    data = data.dropna(subset=["Return"]).reset_index(
        drop=True
    )

    if data.empty:
        raise ValueError(
            "At least two historical prices are required."
        )

    return data
