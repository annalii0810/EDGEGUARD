
import numpy as np
import pytest

from edgeguard.simulator import generate_returns


def test_output_length():
    returns, means = generate_returns(
        n=100,
        change_point=50
    )

    assert len(returns) == 100
    assert len(means) == 100


def test_reproducibility():
    first, _ = generate_returns(seed=42)
    second, _ = generate_returns(seed=42)

    assert np.array_equal(first, second)


def test_change_point():
    _, means = generate_returns(
        n=100,
        change_point=50,
        mu_before=0.01,
        mu_after=-0.01,
    )

    assert np.all(means[:50] == 0.01)
    assert np.all(means[50:] == -0.01)


def test_zero_noise():
    returns, means = generate_returns(sigma=0)

    assert np.array_equal(returns, means)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        generate_returns(n=0)

    with pytest.raises(ValueError):
        generate_returns(change_point=2000, n=1000)

    with pytest.raises(ValueError):
        generate_returns(sigma=-1)
