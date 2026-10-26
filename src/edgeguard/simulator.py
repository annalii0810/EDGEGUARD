
import numpy as np


def generate_returns(
    n=1000,
    change_point=500,
    mu_before=0.001,
    mu_after=-0.001,
    sigma=0.01,
    seed=42,
):
    """
    Generate simulated trading strategy returns.

    The strategy begins with a positive expected return,
    then changes to a negative expected return.
    """

    # Check that the inputs are valid
    if n <= 0:
        raise ValueError("n must be positive")

    if not 0 <= change_point <= n:
        raise ValueError("Invalid change point")

    if sigma < 0:
        raise ValueError("Sigma cannot be negative")

    # Create a reproducible random generator
    rng = np.random.default_rng(seed)

    # Create the true expected returns
    true_mean = np.full(n, mu_before)

    # Change the expected return after the change point
    true_mean[change_point:] = mu_after

    # Generate random market noise
    noise = rng.normal(0, sigma, n)

    # Simulated returns = expected return + noise
    returns = true_mean + noise

    return returns, true_mean
