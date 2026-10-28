
import pandas as pd


def chronological_split(
    data,
    train_end="2018-12-31",
    validation_end="2022-12-31",
):
    """
    Split historical observations chronologically.

    Returns:
        train, validation, test

    The final test period must not be used for
    parameter selection or model development.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a DataFrame.")

    if "Date" not in data.columns:
        raise ValueError("Data must contain a Date column.")

    if data.empty:
        raise ValueError("Data cannot be empty.")

    dates = pd.to_datetime(data["Date"], errors="raise")

    if dates.isna().any():
        raise ValueError("Dates cannot be missing.")

    if dates.duplicated().any():
        raise ValueError("Duplicate dates are not allowed.")

    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)

    if train_end >= validation_end:
        raise ValueError(
            "Training must end before validation."
        )

    ordered = data.copy()
    ordered["Date"] = dates
    ordered = ordered.sort_values(
        "Date"
    ).reset_index(drop=True)

    train = ordered[
        ordered["Date"] <= train_end
    ].copy()

    validation = ordered[
        (ordered["Date"] > train_end)
        & (ordered["Date"] <= validation_end)
    ].copy()

    test = ordered[
        ordered["Date"] > validation_end
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "All three splits must contain observations."
        )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
