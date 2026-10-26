
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from edgeguard.simulator import generate_returns


# Generate 1,000 simulated trading returns
returns, true_mean = generate_returns(
    n=1000,
    change_point=500,
    mu_before=0.001,
    mu_after=-0.001,
    sigma=0.01,
    seed=42
)

# Split returns into two market regimes
before = returns[:500]
after = returns[500:]

# Calculate the average return in each regime
print("EDGEGUARD - Experiment 01")
print("-------------------------")
print(f"Average before: {np.mean(before):.6f}")
print(f"Average after:  {np.mean(after):.6f}")

# Calculate the rolling average of the last 50 returns
window = 50
rolling_mean = np.convolve(
    returns,
    np.ones(window) / window,
    mode="valid"
)

# Create the research graph
plt.figure(figsize=(11, 6))

plt.plot(
    returns,
    alpha=0.25,
    label="Observed Returns"
)

plt.plot(
    np.arange(window - 1, len(returns)),
    rolling_mean,
    label="Rolling Average (50)"
)

plt.plot(
    true_mean,
    linestyle="--",
    label="True Expected Return"
)

plt.axvline(
    x=500,
    color="red",
    linestyle=":",
    label="Regime Change"
)

plt.title("EDGEGUARD: Strategy Edge Deterioration")
plt.xlabel("Trading Opportunity")
plt.ylabel("Net Return")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

# Save the graph
Path("results").mkdir(exist_ok=True)
plt.savefig("results/experiment_01.png", dpi=200)

plt.show()
