
import pandas as pd
import pytest

from edgeguard.data_split import chronological_split


def make_data():
    return pd.DataFrame({
        "Date": pd.to_datetime([
            "2018-12-31",
            "2019-01-02",
            "2020-06-15",
            "2022-12-30",
            "2023-01-03",
            "2024-01-02",
        ]),
        "Return": [0.01, 0.02, -0.01, 0.03, 0.01, -0.02],
    })


def test_correct_split():
    train, validation, test = chronological_split(make_data())

    assert len(train) == 1
    assert len(validation) == 3
    assert len(test) == 2


def test_no_overlapping_dates():
    train, validation, test = chronological_split(make_data())

    train_dates = set(train["Date"])
    validation_dates = set(validation["Date"])
    test_dates = set(test["Date"])

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_chronological_order():
    train, validation, test = chronological_split(make_data())

    assert train["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < test["Date"].min()


def test_unsorted_input():
    data = make_data().sample(frac=1, random_state=42)

    train, validation, test = chronological_split(data)

    for part in (train, validation, test):
        assert part["Date"].is_monotonic_increasing


def test_duplicate_dates():
    data = make_data()
    data.loc[1, "Date"] = data.loc[0, "Date"]

    with pytest.raises(ValueError):
        chronological_split(data)


def test_invalid_boundaries():
    with pytest.raises(ValueError):
        chronological_split(
            make_data(),
            train_end="2023-01-01",
            validation_end="2022-01-01",
        )


def test_empty_split():
    with pytest.raises(ValueError):
        chronological_split(
            make_data(),
            train_end="2010-01-01",
            validation_end="2011-01-01",
        )


def test_missing_date_column():
    data = pd.DataFrame({"Return": [0.01, 0.02]})

    with pytest.raises(ValueError):
        chronological_split(data)
