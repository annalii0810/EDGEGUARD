
import numpy as np
import pandas as pd
import pytest

from edgeguard.backtest import run_backtest


def make_data():
    return pd.DataFrame({
        "Return": [0.10, -0.05, 0.02, 0.03],
        "Exposure": [1.0, 1.0, 0.5, 0.5],
    })


def test_output_length():
    result, metrics = run_backtest(make_data())

    assert len(result) == 4
    assert metrics["observations"] == 4


def test_gross_returns():
    result, _ = run_backtest(
        make_data(), transaction_cost_bps=0
    )

    assert np.allclose(
        result["GrossReturn"],
        [0.10, -0.05, 0.01, 0.015]
    )


def test_transaction_costs():
    result, _ = run_backtest(
        make_data(),
        transaction_cost_bps=10,
        initial_exposure=0,
    )

    assert np.allclose(
        result["Turnover"],
        [1.0, 0.0, 0.5, 0.0]
    )

    assert np.allclose(
        result["TransactionCost"],
        [0.001, 0.0, 0.0005, 0.0]
    )


def test_net_returns():
    result, _ = run_backtest(
        make_data(), transaction_cost_bps=10
    )

    assert np.allclose(
        result["NetReturn"],
        [0.099, -0.05, 0.0095, 0.015]
    )


def test_total_return():
    result, metrics = run_backtest(
        make_data(), transaction_cost_bps=0
    )

    expected = np.prod(
        [1.10, 0.95, 1.01, 1.015]
    ) - 1

    assert np.isclose(
        metrics["total_return"], expected
    )

    assert np.isclose(
        result["Equity"].iloc[-1], 1 + expected
    )


def test_max_drawdown():
    _, metrics = run_backtest(
        make_data(), transaction_cost_bps=0
    )

    assert np.isclose(
        metrics["max_drawdown"], 0.05
    )


def test_no_exposure_change():
    data = pd.DataFrame({
        "Return": [0.01, 0.02, -0.01],
        "Exposure": [0.0, 0.0, 0.0],
    })

    result, _ = run_backtest(data)

    assert np.allclose(result["Turnover"], 0)
    assert np.allclose(result["NetReturn"], 0)


def test_invalid_exposure():
    data = make_data()
    data.loc[1, "Exposure"] = 1.5

    with pytest.raises(ValueError):
        run_backtest(data)


def test_invalid_transaction_cost():
    for value in [-1, np.nan, np.inf, True]:
        with pytest.raises(ValueError):
            run_backtest(
                make_data(),
                transaction_cost_bps=value
            )


def test_invalid_returns():
    data = make_data()
    data.loc[1, "Return"] = -1.0

    with pytest.raises(ValueError):
        run_backtest(data)


def test_invalid_initial_exposure():
    for value in [-0.1, 1.1, np.nan, True]:
        with pytest.raises(ValueError):
            run_backtest(
                make_data(),
                initial_exposure=value
            )


def test_no_future_information():
    data = make_data()

    original, _ = run_backtest(data)

    changed = data.copy()
    changed.loc[3, "Return"] = -0.50

    modified, _ = run_backtest(changed)

    assert np.allclose(
        original["NetReturn"].iloc[:3],
        modified["NetReturn"].iloc[:3]
    )



def test_split_boundary_exit_cost():
    import pandas as pd
    import pytest

    from edgeguard.backtest import run_backtest

    data = pd.DataFrame({
        "Return": [0.01, -0.02],
        "Exposure": [0.0, 0.0],
    })

    result, metrics = run_backtest(
        data,
        transaction_cost_bps=5.0,
        initial_exposure=1.0,
    )

    assert result["Turnover"].iloc[0] == pytest.approx(1.0)
    assert result["TransactionCost"].iloc[0] == pytest.approx(0.0005)
    assert result["NetReturn"].iloc[0] == pytest.approx(-0.0005)
    assert result["Turnover"].iloc[1] == pytest.approx(0.0)
    assert metrics["total_turnover"] == pytest.approx(1.0)
