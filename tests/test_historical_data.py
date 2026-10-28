
import numpy as np
import pandas as pd
import pytest

from edgeguard.historical_data import load_historical_data


def test_correct_returns(tmp_path):
    path = tmp_path / "prices.csv"

    pd.DataFrame({
        "Date": [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04"
        ],
        "Close": [100.0, 110.0, 99.0]
    }).to_csv(path, index=False)

    data = load_historical_data(path)

    assert len(data) == 2
    assert np.allclose(
        data["Return"],
        [0.10, -0.10]
    )


def test_chronological_sorting(tmp_path):
    path = tmp_path / "prices.csv"

    pd.DataFrame({
        "Date": [
            "2024-01-04",
            "2024-01-02",
            "2024-01-03"
        ],
        "Close": [121.0, 100.0, 110.0]
    }).to_csv(path, index=False)

    data = load_historical_data(path)

    assert data["Date"].is_monotonic_increasing
    assert np.allclose(
        data["Return"],
        [0.10, 0.10]
    )


def test_missing_columns(tmp_path):
    path = tmp_path / "prices.csv"

    pd.DataFrame({
        "Date": ["2024-01-02"],
        "Price": [100.0]
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_historical_data(path)


def test_duplicate_dates(tmp_path):
    path = tmp_path / "prices.csv"

    pd.DataFrame({
        "Date": [
            "2024-01-02",
            "2024-01-02"
        ],
        "Close": [100.0, 101.0]
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_historical_data(path)


def test_invalid_prices(tmp_path):
    invalid_values = [
        0.0,
        -10.0,
        np.inf,
        np.nan
    ]

    for value in invalid_values:
        path = tmp_path / "prices.csv"

        pd.DataFrame({
            "Date": [
                "2024-01-02",
                "2024-01-03"
            ],
            "Close": [100.0, value]
        }).to_csv(path, index=False)

        with pytest.raises(ValueError):
            load_historical_data(path)


def test_invalid_dates(tmp_path):
    path = tmp_path / "prices.csv"

    pd.DataFrame({
        "Date": [
            "2024-01-02",
            "not-a-date"
        ],
        "Close": [100.0, 101.0]
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_historical_data(path)


def test_insufficient_prices(tmp_path):
    path = tmp_path / "prices.csv"

    pd.DataFrame({
        "Date": ["2024-01-02"],
        "Close": [100.0]
    }).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_historical_data(path)
