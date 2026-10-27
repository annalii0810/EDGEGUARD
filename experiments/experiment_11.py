
from edgeguard.volatility import rolling_volatility


# Example strategy returns
example_returns = [0.01, 0.02, 0.03, 0.04, 0.05]

# Estimate volatility using historical observations
estimates = rolling_volatility(
    example_returns,
    window=3
)

print("EDGEGUARD - Experiment 11")
print("Historical Volatility Estimation")
print("--------------------------------")
print("Returns:", example_returns)
print("Estimated volatility:", estimates)
