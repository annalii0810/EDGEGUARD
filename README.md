# EDGEGUARD
### Adaptive Statistical Edge Monitoring for Quantitative Trading

**Status:** Week 1 — Research Foundation

## 1. Project Overview

EDGEGUARD is a quantitative trading research and software engineering project designed to investigate how statistical methods can detect deterioration in a trading strategy's expected profitability.

The long-term goal is to develop and evaluate an adaptive risk-control framework that can recognize when a strategy may have lost its statistical advantage.

## 2. Research Question

Can sequential statistical monitoring detect when a quantitative trading strategy's expected net return deteriorates while minimizing false alarms, detection delay, and unnecessary trading interruptions?

## 3. Mathematical Model

We model simulated strategy returns using:

**R(t) = μ(t) + σZ(t)**

Where:

- R(t) represents the observed net return.
- μ(t) represents the true expected net return.
- σ represents return volatility.
- Z(t) represents independent standard normal noise.

The expected return changes at a predefined observation.

### Simulation Parameters

| Parameter | Value |
|---|---|
| Number of observations | 1,000 |
| Change point | 500 |
| Expected return before change | +0.10% |
| Expected return after change | −0.10% |
| Standard deviation | 1.00% |
| Random seed | 42 |

## 4. Experiment 01 — Strategy Edge Deterioration

The first experiment simulates a strategy transitioning from positive to negative expected returns.

A 50-observation rolling average is used to visualize how estimates of profitability respond to the regime change.

### Experimental Results

| Measurement | Observed result |
|---|---|
| Mean return before change | +0.0897% |
| Mean return after change | −0.1447% |
| Automated unit tests | 5 passed |

The observed averages differ from the true expected returns because of random variation.

### Visualization

![Experiment 01](results/experiment_01.png)

## 5. Current Limitations

- Returns are generated from a simplified Gaussian distribution.
- Observations are independent.
- The regime change is artificially specified.
- No live trading or historical market data is used.
- No formal change-detection algorithm has been evaluated yet.
- The experiment does not establish real-world trading profitability.

## 6. Technologies

Python, NumPy, Matplotlib, pytest, Git, and GitHub.

## 7. Next Steps

1. Implement a rolling-mean detection baseline.
2. Measure false alarms and detection delay.
3. Compare statistical detection methods.
4. Introduce more realistic market assumptions.
5. Develop and evaluate an adaptive risk-control framework.

## 8. Reproducibility

Install the dependencies:

`python -m pip install -r requirements.txt`

Run automated tests:

`PYTHONPATH=src python -m pytest -v`

Run Experiment 01:

`PYTHONPATH=src python experiments/experiment_01.py`

All commands should be run from the project root directory.