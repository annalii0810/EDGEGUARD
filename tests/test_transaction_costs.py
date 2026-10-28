
import numpy as np
import pytest

from edgeguard.transaction_costs import calculate_transaction_costs


def test_exposure_reduction_cost():
    exposures = [1.0, 1.0, 1.0, 0.25, 0.25]

    costs = calculate_transaction_costs(exposures)

    expected = [0.0, 0.0, 0.0, 0.00075, 0.0]

    assert np.allclose(costs, expected)


def test_no_exposure_change():
    exposures = [1.0, 1.0, 1.0, 1.0]

    costs = calculate_transaction_costs(exposures)

    assert np.allclose(costs, np.zeros(4))


def test_full_exit():
    exposures = [1.0, 1.0, 0.0, 0.0]

    costs = calculate_transaction_costs(exposures)

    assert np.allclose(
        costs,
        [0.0, 0.0, 0.001, 0.0]
    )


def test_initial_entry_cost():
    exposures = [1.0, 1.0, 1.0]

    costs = calculate_transaction_costs(
        exposures,
        initial_exposure=0.0
    )

    assert np.allclose(
        costs,
        [0.001, 0.0, 0.0]
    )


def test_custom_transaction_costs():
    exposures = [1.0, 0.5, 0.5]

    costs = calculate_transaction_costs(
        exposures,
        commission_bps=2.0,
        slippage_bps=3.0
    )

    assert np.allclose(
        costs,
        [0.0, 0.00025, 0.0]
    )


def test_invalid_exposures():
    invalid = [
        [-0.1, 0.5],
        [1.2, 0.5],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [[1.0, 0.5]],
    ]

    for exposures in invalid:
        with pytest.raises(ValueError):
            calculate_transaction_costs(exposures)


def test_invalid_cost_parameters():
    for value in [-1.0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            calculate_transaction_costs(
                [1.0, 0.5],
                commission_bps=value
            )

        with pytest.raises(ValueError):
            calculate_transaction_costs(
                [1.0, 0.5],
                slippage_bps=value
            )


def test_invalid_initial_exposure():
    for value in [-0.1, 1.5, np.nan]:
        with pytest.raises(ValueError):
            calculate_transaction_costs(
                [1.0, 0.5],
                initial_exposure=value
            )

