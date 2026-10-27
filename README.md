# EDGEGUARD

**Sequential Statistical Monitoring for Quantitative Trading Strategies**

*A reproducible quantitative research project investigating statistical detection of deteriorating trading strategy profitability.*

## 1. Project Overview

Quantitative trading strategies can lose their statistical advantage as market conditions evolve. However, distinguishing genuine deterioration from ordinary random fluctuations is challenging.

A strategy may experience temporary losses while remaining profitable in expectation. Conversely, a deteriorating strategy may continue producing occasional positive returns.

**EDGEGUARD** investigates how sequential statistical monitoring can identify deterioration while balancing three competing objectives:

- Minimizing false alarms.
- Detecting genuine deterioration quickly.
- Avoiding missed changes.

The project implements and evaluates rolling-mean monitoring and cumulative-sum (CUSUM) detection using reproducible synthetic return simulations.

**Current status:** Initial simulation framework, rolling-mean detector, CUSUM detector, threshold calibration, and synthetic benchmarking implemented.

## 2. Research Question

**Can sequential statistical change-detection methods identify deterioration in a quantitative trading strategy's expected net return while controlling false alarms and minimizing detection delay?**

The research investigates the trade-off between detection sensitivity, reliability, and responsiveness.

## 3. Mathematical Framework

### 3.1 Synthetic Return Model

We model strategy returns as:

$$
R_t = \mu_t + \sigma Z_t
$$

Where:

- $R_t$ represents the observed strategy return at observation $t$.
- $\mu_t$ represents the true expected return.
- $\sigma$ represents return standard deviation.
- $Z_t$ represents independent standard normal noise.

We assume:

$$
Z_t \sim \mathcal{N}(0,1)
$$

### 3.2 Regime-Change Model

The strategy initially has a positive expected return. At a predefined change point, the expected return becomes negative.

$$
\mu_t =
\begin{cases}
0.001, & t < 500 \\
-0.001, & t \geq 500
\end{cases}
$$

The simulation uses:

| Parameter | Value |
|---|---|
| Number of observations | 1,000 |
| Change-point index | 500 |
| Expected return before change | +0.10% |
| Expected return after change | -0.10% |
| Return standard deviation | 1.00% |
| Initial random seed | 42 |

These assumptions provide a controlled environment where the true change point is known for evaluation.

The detector does not receive the true change point as an input.

## 4. Statistical Detection Algorithms

### 4.1 Rolling-Mean Detector

The rolling-mean detector estimates recent strategy profitability using a fixed observation window.

For a window containing $w$ observations:

$$
\bar{R}_t = \frac{1}{w}\sum_{i=t-w+1}^{t}R_i
$$

Where:

- $\bar{R}_t$ is the rolling average.
- $w$ is the rolling-window size.
- $R_i$ is the observed return at index $i$.

The initial detection rule is:

$$
\bar{R}_t < 0
$$

An alarm occurs when the recent average return becomes negative.

This simple rule is sensitive to random fluctuations and can produce frequent false alarms.

#### Statistical Threshold Adjustment

To investigate more conservative detection, we evaluate:

$$
\bar{R}_t < -k\frac{\sigma}{\sqrt{w}}
$$

Where $k$ controls the alarm sensitivity.

Larger values of $k$ require stronger negative evidence before triggering an alarm.

These thresholds are heuristic and do not automatically provide a valid sequential significance guarantee.

### 4.2 CUSUM Detector

CUSUM, or *Cumulative Sum*, is a sequential change-detection method.

Unlike a rolling mean, CUSUM accumulates evidence of sustained deviations from a reference level.

The downward CUSUM statistic is:

$$
S_t = \max\left(0,\ S_{t-1}+\mu_0-R_t-\delta\right)
$$

With initial condition:

$$
S_{-1}=0
$$

Where:

- $S_t$ is the accumulated detection statistic.
- $\mu_0$ is the reference expected return.
- $R_t$ is the current observed return.
- $\delta$ is a nonnegative allowance parameter.

The detector triggers when:

$$
S_t > h
$$

Where $h$ is the alarm threshold.

CUSUM accumulates evidence across observations, potentially improving sensitivity to sustained changes.

Both rolling-mean monitoring and CUSUM are established statistical techniques. Their implementation alone does not constitute a novel research contribution.

## 5. Experimental Methodology

### 5.1 Reproducibility

Experiments use NumPy's seeded random-number generator to produce reproducible synthetic returns.

Multiple independent simulation sequences are generated using different random seeds.

### 5.2 Alarm Classification

For simulations containing a known change point $\tau$, we classify the first alarm into one of three mutually exclusive outcomes.

**Premature alarm:** The first alarm occurs before $\tau$.

**Successful detection:** The first alarm occurs at or after $\tau$.

**Missed detection:** No alarm occurs during the observation horizon.

The current evaluation uses a stop-on-first-alarm policy.

### 5.3 Detection Delay

For successful detections, delay is measured as:

$$
D = t_{\mathrm{alarm}}-\tau
$$

Where:

- $t_{\mathrm{alarm}}$ is the first alarm index.
- $\tau$ is the true change-point index.

Average delay is calculated only among successfully detected simulations.

This conditional metric must be interpreted alongside premature-alarm and missed-detection proportions.

### 5.4 False-Alarm Probability

Under a no-change scenario, the expected return remains positive throughout the simulation.

We estimate:

$$
\widehat{P}_{FA}=\frac{N_{FA}}{N_{\mathrm{sim}}}
$$

Where:

- $N_{FA}$ is the number of simulations containing at least one alarm.
- $N_{\mathrm{sim}}$ is the total number of simulations.

This measures the probability of at least one false alarm over a fixed monitoring horizon, not the per-observation false-alarm rate.

## 6. Experimental Results

### Experiment 01 — Synthetic Strategy Edge Deterioration

The first experiment generated 1,000 synthetic returns with a regime change at index 500.

Observed average returns:

| Measurement | Result |
|---|---:|
| Average before change | +0.0869% |
| Average after change | -0.1447% |

The observed averages differ from the true expected returns because of random variation.

![Synthetic Strategy Edge Deterioration](results/experiment_01.png)

### Experiment 02 — Initial Rolling-Mean Detection

The initial rolling-mean detector used:

- Window size: 50 observations.
- Alarm threshold: zero.
- Number of simulations: 500.

Results:

| Outcome | Count |
|---|---:|
| Premature alarms | 498 |
| Successful detections | 2 |
| Missed detections | 0 |

The premature-alarm proportion was 99.6%.

This demonstrates the sensitivity of a zero-threshold rolling-mean detector to noisy returns under the tested assumptions.

### Experiment 03 — Rolling-Window Comparison

Four rolling-window sizes were evaluated.

| Window | Premature alarms | Successful detections | Conditional average delay |
|---|---:|---:|---:|
| 10 | 100.0% | 0.0% | N/A |
| 50 | 99.6% | 0.4% | 50.00 |
| 100 | 89.6% | 10.4% | 50.15 |
| 200 | 42.2% | 57.8% | 97.76 |

Larger windows reduced premature alarms but changed detection responsiveness.

### Experiment 04 — No-Change False-Alarm Evaluation

The strategy maintained a constant positive expected return throughout 1,000 observations.

| Window | False-alarm proportion |
|---|---:|
| 10 | 100.0% |
| 50 | 100.0% |
| 100 | 98.6% |
| 200 | 75.6% |

Even the longest tested window generated frequent false alarms over the monitoring horizon.

### Experiment 05 — Statistical Threshold Comparison

The rolling-mean detector was evaluated using multiple statistical thresholds.

| Threshold | Premature alarms | Successful detections | Missed detections | Average delay |
|---|---:|---:|---:|---:|
| $k=0$ | 99.6% | 0.4% | 0.0% | 50.00 |
| $k=1$ | 78.8% | 21.2% | 0.0% | 61.08 |
| $k=2$ | 16.4% | 80.4% | 3.2% | 159.43 |
| $k=3$ | 0.8% | 47.2% | 52.0% | 248.56 |

Increasing the threshold strictness reduced premature alarms but increased detection delays and, at the strictest setting, missed detections.

### Experiment 06 — Calibrated Rolling-Mean Detection

The rolling-mean threshold was calibrated using 1,000 no-change simulations.

The target was approximately 5% false alarms over a 1,000-observation monitoring horizon.

The calibrated threshold was:

$$
h_{\mathrm{RM}}=-0.003852
$$

Independent evaluation results:

| Metric | Result |
|---|---:|
| No-change false alarms | 6.6% |
| Premature alarms | 3.8% |
| Successful detections | 59.6% |
| Missed detections | 36.6% |
| Conditional average delay | 213.17 |

Threshold calibration substantially reduced false alarms compared with the original zero-threshold detector, but deterioration detection remained relatively slow.

### Experiment 07 — Calibrated CUSUM Detection

CUSUM was calibrated independently using 1,000 no-change simulations.

The calibrated CUSUM threshold was:

$$
h_{\mathrm{CUSUM}}=0.273950
$$

Independent evaluation results:

| Metric | Result |
|---|---:|
| No-change false alarms | 6.8% |
| Premature alarms | 2.8% |
| Successful detections | 93.8% |
| Missed detections | 3.4% |
| Conditional average delay | 195.42 |

CUSUM detected deterioration in 469 of 500 regime-change simulations.

## 7. Calibrated Detector Comparison

| Metric | Rolling Mean | CUSUM |
|---|---:|---:|
| No-change false alarms | 6.6% | 6.8% |
| Premature alarms | 3.8% | 2.8% |
| Successful detections | 59.6% | 93.8% |
| Missed detections | 36.6% | 3.4% |
| Conditional average delay | 213.17 | 195.42 |

Under the tested synthetic conditions, CUSUM achieved a 34.2-percentage-point higher successful-detection proportion than the calibrated rolling-mean detector.

However, the average detection delays are conditional on different subsets of successful simulations.

The results provide preliminary evidence favoring CUSUM under the specified model, not proof of superiority across different market conditions.

## 8. Repository Structure

```text
EDGEGUARD/
├── src/
│   └── edgeguard/
│       ├── __init__.py
│       ├── simulator.py
│       └── detectors.py
├── tests/
│   ├── test_simulator.py
│   └── test_detectors.py
├── experiments/
│   ├── experiment_01.py
│   ├── experiment_02.py
│   ├── experiment_03.py
│   ├── experiment_04.py
│   ├── experiment_05.py
│   ├── experiment_06.py
│   └── experiment_07.py
├── results/
│   └── experiment_01.png
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Installation and Reproducibility

### Requirements

- Python 3
- NumPy
- Matplotlib
- pytest

### Setup

Clone the repository:

```bash
git clone https://github.com/annalii0810/EDGEGUARD.git
cd EDGEGUARD
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run Automated Tests

```bash
PYTHONPATH=src python -m pytest -v
```

The current implementation has 15 passing automated tests.

### Run Experiments

```bash
PYTHONPATH=src python experiments/experiment_01.py
PYTHONPATH=src python experiments/experiment_02.py
PYTHONPATH=src python experiments/experiment_03.py
PYTHONPATH=src python experiments/experiment_04.py
PYTHONPATH=src python experiments/experiment_05.py
PYTHONPATH=src python experiments/experiment_06.py
PYTHONPATH=src python experiments/experiment_07.py
```

## 10. Research Limitations

The current experimental framework has several important limitations.

- Returns are independent and Gaussian.
- Expected returns and volatility are specified by the simulator.
- Only a limited set of regime-change scenarios has been evaluated.
- Calibration targets a fixed observation horizon.
- Current detector comparisons do not include formal uncertainty intervals.
- Detector parameters have not been optimized across a broad validation set.
- Historical market data has not yet been evaluated.
- Transaction costs, slippage, and execution effects are not modeled separately.
- The experiments do not establish real-world trading profitability.

## 11. Future Research Roadmap

### Phase 1 — Statistical Foundations

- [x] Implement synthetic return generator.
- [x] Implement rolling-mean detector.
- [x] Implement CUSUM detector.
- [x] Develop automated unit tests.
- [x] Calibrate detector thresholds.
- [x] Conduct initial synthetic benchmarking.

### Phase 2 — Robustness Evaluation

- [ ] Test different volatility regimes.
- [ ] Evaluate smaller changes in expected returns.
- [ ] Introduce gradual deterioration.
- [ ] Simulate autocorrelated and heavy-tailed returns.
- [ ] Calculate confidence intervals for performance metrics.
- [ ] Evaluate detector performance across multiple independent test sets.

### Phase 3 — Adaptive Risk Control

- [ ] Develop exposure-reduction and pause decisions.
- [ ] Model costs of false alarms and delayed detection.
- [ ] Evaluate alternative risk-control policies.
- [ ] Compare risk-adjusted performance under controlled simulations.

### Phase 4 — Historical Validation

- [ ] Integrate historical market data.
- [ ] Implement chronological out-of-sample evaluation.
- [ ] Model transaction costs and slippage.
- [ ] Evaluate performance under changing market conditions.

### Phase 5 — Research Portfolio

- [ ] Develop an interactive research dashboard.
- [ ] Produce a formal technical research report.
- [ ] Create publication-quality figures.
- [ ] Document reproducible experiments and failure cases.
- [ ] Prepare evidence-based resume descriptions.

## 12. Research Integrity

EDGEGUARD is an educational quantitative research project.

All currently reported results are derived from synthetic simulations and are limited to their specified assumptions.

The project does not claim to demonstrate profitable live trading, guarantee statistical performance in real markets, or introduce a novel change-detection algorithm.

Its long-term research objective is to investigate whether carefully evaluated statistical monitoring can support more reliable quantitative trading risk-control decisions.