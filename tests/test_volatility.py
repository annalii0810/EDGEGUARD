
import numpy as np
import pytest

from edgeguard.volatility import rolling_volatility


def test_rolling_volatility_values():
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]

    estimates = rolling_volatility(
        returns,
        window=3
    )

    assert len(estimates) == 5
    assert np.all(np.isnan(estimates[:3]))
    assert np.allclose(estimates[3:], [0.01, 0.01])



def test_rolling_volatility_no_future_information():
    original = [0.01, 0.02, 0.03, 0.04, 0.05]
    modified = [0.01, 0.02, 0.03, 0.04, 0.50]

    a = rolling_volatility(original, window=3)
    b = rolling_volatility(modified, window=3)

    assert np.allclose(a, b, equal_nan=True)



def test_rolling_volatility_invalid_inputs():

    # Empty return sequence
    with pytest.raises(ValueError):
        rolling_volatility([], window=3)

    # Missing or undefined return
    with pytest.raises(ValueError):
        rolling_volatility([0.01, np.nan, 0.03], window=2)

    # Window too small
    with pytest.raises(ValueError):
        rolling_volatility([0.01, 0.02, 0.03], window=1)

    # Window too large
    with pytest.raises(ValueError):
        rolling_volatility([0.01, 0.02, 0.03], window=4)

    # Window equal to sequence length
    with pytest.raises(ValueError):
        rolling_volatility([0.01, 0.02, 0.03], window=3)

    # Window must be an integer
    with pytest.raises(ValueError):
        rolling_volatility([0.01, 0.02, 0.03], window=2.5)

    # Boolean is not a valid window
    with pytest.raises(ValueError):
        rolling_volatility([0.01, 0.02, 0.03], window=True)



def test_rolling_volatility_zero_variance():
    returns = [0.01, 0.01, 0.01, 0.01, 0.01]

    estimates = rolling_volatility(
        returns,
        window=3
    )

    assert np.all(np.isnan(estimates[:3]))
    assert np.allclose(estimates[3:], [0.0, 0.0])
