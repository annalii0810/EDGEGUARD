
import numpy as np
import pandas as pd
import pytest

from edgeguard.historical_strategy import moving_average_strategy


def make_data():
    prices = np.array([
        100.0, 101.0, 102.0, 103.0,
        104.0, 105.0, 104.0, 103.0,
        102.0, 101.0
    ])

    returns = np.zeros(len(prices))
    returns[1:] = prices[1:] / prices[:-1] - 1

    return pd.DataFrame({
        "Date": pd.date_range(
            "2024-01-01",
            periods=len(prices),
            freq="D"
        ),
        "Close": prices,
        "Return": returns
    })


def test_output_length():
    data = make_data()
    result = moving_average_strategy(data, window=3)

    assert len(result) == len(data)


def test_initial_exposure():
    result = moving_average_strategy(
        make_data(), window=3
    )

    assert np.all(result["Exposure"].iloc[:3] == 0)


def test_strategy_returns():
    result = moving_average_strategy(
        make_data(), window=3
    )

    expected = result["Exposure"] * result["Return"]

    assert np.allclose(
        result["StrategyReturn"], expected
    )


def test_no_future_information():
    data = make_data()

    original = moving_average_strategy(data, window=3)

    modified = data.copy()
    modified.loc[7:, "Close"] *= 2

    modified["Return"] = (
        modified["Close"].pct_change().fillna(0)
    )

    changed = moving_average_strategy(
        modified, window=3
    )

    # Changing future prices cannot change earlier decisions.
    assert np.allclose(
        original["Exposure"].iloc[:7],
        changed["Exposure"].iloc[:7]
    )

    assert np.allclose(
        original["StrategyReturn"].iloc[:7],
        changed["StrategyReturn"].iloc[:7]
    )


def test_invalid_window():
    for window in [0, 1, -3, 2.5, True]:
        with pytest.raises(ValueError):
            moving_average_strategy(
                make_data(), window=window
            )


def test_missing_columns():
    data = make_data().drop(columns=["Close"])

    with pytest.raises(ValueError):
        moving_average_strategy(data)


def test_unsorted_dates():
    data = make_data().iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError):
        moving_average_strategy(data)


def test_invalid_prices():
    data = make_data()
    data.loc[3, "Close"] = -100.0

    with pytest.raises(ValueError):
        moving_average_strategy(data)


def test_duplicate_dates():
    data = make_data()
    data.loc[2, "Date"] = data.loc[1, "Date"]

    with pytest.raises(ValueError):
        moving_average_strategy(data)


def test_inconsistent_returns():
    data = make_data()

    # Deliberately corrupt a return without changing prices.
    data.loc[5, "Return"] = 0.50

    with pytest.raises(
        ValueError,
        match="Return values do not match"
    ):
        moving_average_strategy(data, window=3)
