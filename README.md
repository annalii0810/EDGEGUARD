
# EDGEGUARD

## Sequential Statistical Monitoring for Quantitative Trading Strategies

*A reproducible Python research project investigating how sequential change-detection methods identify deterioration in simulated trading strategy profitability.*

## 1. Project Overview

Quantitative trading strategies can lose their statistical advantage as market conditions change. Temporary losses do not necessarily mean that a strategy has deteriorated, while a deteriorating strategy may still occasionally produce positive returns.

EDGEGUARD investigates the trade-offs among false alarms, successful detection, missed changes, and detection delay.

The project implements:

- Rolling-mean monitoring
- Downward cumulative sum (CUSUM) detection
- Historical volatility estimation
- Volatility-normalized adaptive CUSUM
- Independent calibration and evaluation
- Reproducible Monte Carlo simulations

**Current status:** Experiments 01–13 completed, with 25 automated tests passing in the latest full test run. Phase 2 implementation and baseline comparisons are complete. Broader robustness validation remains future work.

## 2. Research Question

Can sequential statistical change-detection methods identify deterioration in a quantitative trading strategy's expected return while controlling false alarms and minimizing detection delay?

A related question is whether estimating volatility causally from historical returns can provide performance comparable to a detector that knows the true simulated volatility.

## 3. Mathematical Framework

### 3.1 Synthetic Return Model

Strategy returns are modeled as:

```math
R_t = \mu_t + \sigma Z_t
```

where:

- $R_t$ is the simulated strategy return at time $t$.
- $\mu_t$ is the expected return at time $t$.
- $\sigma$ is the return volatility.
- $Z_t$ is independent standard normal noise.

The noise follows:

```math
Z_t \sim \mathcal{N}(0,1)
```

### 3.2 Regime-Change Model

The expected return changes according to:

```math
\mu_t =
\begin{cases}
0.001, & t < 500 \\
-0.001, & t \ge 500
\end{cases}
```

where:

- $\mu_t$ is the expected return at time $t$.
- $\tau = 500$ is the simulated change point.
- $\mu_{\text{before}} = 0.001$ is the expected return before deterioration.
- $\mu_{\text{after}} = -0.001$ is the expected return after deterioration.

| Parameter | Value |
|---|---:|
| Observations per simulation | 1,000 |
| Change point ($\tau$) | 500 |
| Expected return before change | +0.10% |
| Expected return after change | -0.10% |
| Baseline volatility ($\sigma$) | 1.00% |

These are simplified synthetic assumptions rather than a realistic model of live trading profitability.

## 4. Statistical Detection Algorithms

### 4.1 Rolling-Mean Detector

The rolling mean is:

```math
\bar{R}_t =
\frac{1}{w}
\sum_{i=t-w+1}^{t} R_i
```

where:

- $\bar{R}_t$ is the rolling average return.
- $w$ is the rolling-window size.
- $R_i$ is the observed return at time $i$.

An alarm occurs when:

```math
\bar{R}_t < h
```

where $h$ is the detection threshold.

The original experiment used $h = 0$. Later experiments calibrated the threshold using simulated no-change sequences.

### 4.2 Downward CUSUM Detector

The downward cumulative sum is defined as:

```math
S_t =
\max\left(
0,\,
S_{t-1}+\mu_0-R_t-\delta
\right)
```

with initial condition:

```math
S_{-1}=0
```

where:

- $S_t$ is the cumulative detection statistic.
- $S_{t-1}$ is the previous cumulative statistic.
- $\mu_0$ is the target expected return.
- $R_t$ is the observed return.
- $\delta$ is the allowance parameter.

An alarm occurs when:

```math
S_t > h
```

where $h$ is the calibrated alarm threshold.

### 4.3 Historical Volatility Estimation

EDGEGUARD estimates volatility using previously observed returns.

The rolling sample standard deviation is:

```math
\hat{\sigma}_t =
\sqrt{
\frac{1}{w-1}
\sum_{i=t-w}^{t-1}
(R_i-\bar{R}_{t-1})^2
}
```

where:

```math
\bar{R}_{t-1} =
\frac{1}{w}
\sum_{i=t-w}^{t-1}R_i
```

and:

- $\hat{\sigma}_t$ is the estimated volatility at time $t$.
- $w$ is the historical estimation window.
- $\bar{R}_{t-1}$ is the mean of the preceding $w$ returns.

Only returns observed before time $t$ are used to estimate volatility for time $t$.

This avoids look-ahead bias in the volatility estimator.

### 4.4 Adaptive CUSUM Detector

Adaptive CUSUM normalizes return deviations using estimated historical volatility:

```math
A_t =
\max\left(
0,\,
A_{t-1}
+
\frac{\mu_0-R_t}
{\max(\hat{\sigma}_t,\sigma_{\min})}
-k
\right)
```

where:

- $A_t$ is the adaptive CUSUM statistic.
- $A_{t-1}$ is the previous cumulative statistic.
- $\mu_0$ is the target expected return.
- $R_t$ is the observed return.
- $\hat{\sigma}_t$ is the historical volatility estimate.
- $\sigma_{\min}$ is the minimum permitted volatility.
- $k$ is the standardized allowance.

The effective volatility is:

```math
\sigma_{\text{effective},t}
=
\max(\hat{\sigma}_t,\sigma_{\min})
```

The volatility floor prevents division by zero and limits excessive amplification of small return deviations.

An alarm occurs when:

```math
A_t > h
```

where $h$ is the calibrated detection threshold.

## 5. Evaluation Methodology

### 5.1 Monte Carlo Simulations

Experiments use seeded synthetic Gaussian return sequences.

Different random seeds are used for calibration and independent evaluation.

The same independent test sequences are reused across detectors when performing paired comparisons.

### 5.2 Alarm Classification

The first alarm is classified as:

- **Premature alarm:** The first alarm occurs before the true regime-change point.
- **Successful detection:** The first alarm occurs at or after the regime-change point.
- **Missed detection:** No alarm occurs before the simulation ends.

Each simulation contributes to exactly one of these categories.

### 5.3 False-Alarm Probability

The empirical false-alarm probability is:

```math
\widehat{P}_{FA}
=
\frac{N_{FA}}{N_{\text{sim}}}
```

where:

- $N_{FA}$ is the number of simulations containing a false alarm.
- $N_{\text{sim}}$ is the total number of simulations.

### 5.4 Detection Delay

Detection delay is calculated as:

```math
D = t_{\text{alarm}}-\tau
```

where:

- $D$ is the detection delay.
- $t_{\text{alarm}}$ is the first alarm time.
- $\tau$ is the true regime-change point.

Average detection delay is calculated only among successful detections.

### 5.5 Threshold Calibration

Thresholds are selected from maximum detector statistics observed on no-change calibration simulations.

For a target false-alarm probability of $\alpha = 0.05$:

```math
h =
Q_{1-\alpha}
(M_1,\ldots,M_n)
```

where:

- $h$ is the calibrated threshold.
- $Q_{1-\alpha}$ is the empirical quantile.
- $M_i$ is the maximum detector statistic in calibration simulation $i$.
- $\alpha$ is the target false-alarm probability.

The resulting threshold is then evaluated on independent simulations.

## 6. Experimental Results

### 6.1 Experiments 01–07: Statistical Foundations

| Experiment | Main finding |
|---|---|
| 01 | Simulated deterioration from positive to negative expected returns |
| 02 | Zero-threshold rolling mean produced 99.6% premature alarms |
| 03 | Larger rolling windows reduced premature alarms |
| 04 | No-change experiments exposed excessive false alarms |
| 05 | Stronger thresholds reduced false alarms but increased missed detections |
| 06 | Calibrated rolling mean achieved 59.6% successful detections |
| 07 | Calibrated CUSUM achieved 93.8% successful detections |

#### Experiment 06: Rolling-Mean Calibration

| Metric | Result |
|---|---:|
| Calibrated threshold | -0.003852 |
| No-change false-alarm proportion | 6.6% |
| Premature alarms | 3.8% |
| Successful detections | 59.6% |
| Missed detections | 36.6% |
| Average detection delay | 213.17 |

#### Experiment 07: CUSUM Calibration

| Metric | Result |
|---|---:|
| Calibrated threshold | 0.273950 |
| No-change false-alarm proportion | 6.8% |
| Premature alarms | 2.8% |
| Successful detections | 93.8% |
| Missed detections | 3.4% |
| Average detection delay | 195.42 |

These results motivated further investigation of sequential detection methods.

### 6.2 Experiment 08: Volatility Robustness

Rolling-mean and CUSUM detectors were evaluated under four volatility conditions:

```math
\sigma \in
\{0.005,\ 0.010,\ 0.020,\ 0.030\}
```

Both fixed-threshold methods experienced substantial performance deterioration under increased volatility.

This motivated investigating volatility-normalized detection methods.

### 6.3 Experiment 09: Volatility-Normalized CUSUM

The experiment normalized return deviations using known simulated volatility.

| Volatility | Successful detections |
|---|---:|
| 0.005 | 87.4% |
| 0.010 | 87.2% |
| 0.020 | 61.0% |
| 0.030 | 40.2% |

Premature alarms were 12.6% across all four conditions.

This experiment used true simulated volatility and therefore represents an oracle-style benchmark rather than a directly deployable detector.

### 6.4 Experiment 10: Normalized CUSUM Calibration

The normalized CUSUM detector was calibrated using no-change simulations.

| Metric | Result |
|---|---:|
| Calibrated threshold | 27.394951 |
| Independent no-change false-alarm proportion | 6.8% |

The identical rates across volatility conditions were consistent with the Gaussian scaling assumptions and reuse of standardized random sequences.

### 6.5 Experiment 11: Historical Volatility Estimation

Implemented a causal rolling sample-standard-deviation estimator.

Automated tests verify:

- Correct calculations
- No future-information leakage
- Invalid input handling
- Zero-variance behavior

The estimator was integrated into EDGEGUARD's reusable source code.

### 6.6 Experiment 12: Adaptive CUSUM Calibration

Three volatility floors were evaluated.

| Volatility floor | Calibrated threshold | False-alarm proportion |
|---|---:|---:|
| 0.000001 | 28.233946 | 6.2% |
| 0.000500 | 28.233946 | 6.2% |
| 0.001000 | 28.233946 | 6.2% |

All three volatility floors produced identical results under the tested conditions.

This does not establish which floor is optimal.

#### Adaptive CUSUM Detection Performance

| Volatility | Successful detections | Missed detections | Average delay |
|---|---:|---:|---:|
| 0.005 | 98.0% | 0.0% | 78.75 |
| 0.010 | 94.6% | 3.4% | 193.98 |
| 0.020 | 43.6% | 54.4% | 267.10 |
| 0.030 | 24.0% | 74.0% | 263.78 |

Premature alarms were 2.0% across all four conditions.

### 6.7 Experiment 13: Original vs Adaptive CUSUM

The final Phase 2 experiment compared:

1. Original CUSUM using true simulated volatility to scale its allowance.
2. Adaptive CUSUM using causal historical volatility estimates.

Both detectors were calibrated separately and evaluated on identical independent test sequences.

#### Calibrated Thresholds

| Volatility | Original CUSUM | Adaptive CUSUM |
|---|---:|---:|
| 0.005 | 0.136975 | 28.233946 |
| 0.010 | 0.273950 | 28.233946 |
| 0.020 | 0.547899 | 28.233946 |
| 0.030 | 0.821849 | 28.233946 |

The original CUSUM threshold scales with volatility because the statistic is measured in raw return units.

The adaptive threshold remains constant in these experiments because the statistic uses volatility-normalized deviations.

#### Independent False-Alarm Comparison

| Volatility | Original CUSUM | Adaptive CUSUM |
|---|---:|---:|
| 0.005 | 6.8% | 6.2% |
| 0.010 | 6.8% | 6.2% |
| 0.020 | 6.8% | 6.2% |
| 0.030 | 6.8% | 6.2% |

The adaptive detector generated 31 false alarms out of 500 independent no-change simulations, compared with 34 for the original detector.

This difference is small and does not establish statistical superiority.

#### Successful Detection Comparison

| Volatility | Original CUSUM | Adaptive CUSUM |
|---|---:|---:|
| 0.005 | 97.2% | 98.0% |
| 0.010 | 93.8% | 94.6% |
| 0.020 | 41.8% | 43.6% |
| 0.030 | 24.6% | 24.0% |

#### Missed Detection Comparison

| Volatility | Original CUSUM | Adaptive CUSUM |
|---|---:|---:|
| 0.005 | 0.0% | 0.0% |
| 0.010 | 3.4% | 3.4% |
| 0.020 | 55.4% | 54.4% |
| 0.030 | 72.6% | 74.0% |

#### Conditional Average Detection Delays

| Volatility | Original CUSUM | Adaptive CUSUM |
|---|---:|---:|
| 0.005 | 77.80 | 78.75 |
| 0.010 | 195.42 | 193.98 |
| 0.020 | 262.39 | 267.10 |
| 0.030 | 267.41 | 263.78 |

### 6.8 Interpretation of Final Results

Adaptive CUSUM achieved broadly comparable performance to an oracle benchmark that knows the true simulated volatility.

The adaptive detector achieved slightly higher successful-detection rates at three volatility levels and slightly lower performance at the highest volatility level.

However, the observed differences were small.

Confidence intervals and paired statistical tests have not yet established statistically significant superiority.

Detection became substantially harder as volatility increased relative to the fixed change in expected return.

The results support further investigation of causal volatility normalization but do not demonstrate that adaptive CUSUM is universally superior.

## 7. Repository Structure

```text
EDGEGUARD/
├── .vscode/
│   └── settings.json
├── src/
│   └── edgeguard/
│       ├── __init__.py
│       ├── simulator.py
│       ├── detectors.py
│       └── volatility.py
├── tests/
│   ├── test_simulator.py
│   ├── test_detectors.py
│   └── test_volatility.py
├── experiments/
│   ├── experiment_01.py
│   ├── experiment_02.py
│   ├── experiment_03.py
│   ├── experiment_04.py
│   ├── experiment_05.py
│   ├── experiment_06.py
│   ├── experiment_07.py
│   ├── experiment_08.py
│   ├── experiment_09.py
│   ├── experiment_10.py
│   ├── experiment_11.py
│   ├── experiment_12.py
│   └── experiment_13.py
├── results/
│   └── experiment_01.png
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Installation and Reproducibility

### 8.1 Clone the Repository

```bash
git clone https://github.com/annalii0810/EDGEGUARD.git
cd EDGEGUARD
```

### 8.2 Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 8.3 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 8.4 Run Automated Tests

```bash
PYTHONPATH=src python -m pytest -v
```

The latest complete test run passed all 25 automated tests.

### 8.5 Run Experiments

For example:

```bash
PYTHONPATH=src python experiments/experiment_13.py
```

Other experiments can be executed by replacing the experiment filename.

The experiments use seeded random-number generation to support reproducibility.

## 9. Research Limitations

The current project has several important limitations.

- Returns are synthetic, independent, and Gaussian.
- Expected returns and volatility are specified by the simulator.
- The original CUSUM oracle benchmark uses true simulated volatility.
- Some volatility comparisons reuse standardized random sequences.
- Only a limited set of regime-change scenarios has been tested.
- Detection delay is conditional on successful detection.
- Confidence intervals and formal paired statistical tests remain future work.
- Volatility floors and detector parameters are not proven optimal.
- Autocorrelation, heavy tails, and volatility clustering have not yet been evaluated.
- Historical market data and live execution are not included.
- Transaction costs, slippage, and real-world profitability have not been established.

The results should be interpreted as controlled simulation findings, not evidence of profitable live trading.

## 10. Research Roadmap

### Phase 1 — Statistical Foundations

- [x] Synthetic return simulator
- [x] Rolling-mean detector
- [x] Original CUSUM detector
- [x] Initial calibration and benchmarking

### Phase 2 — Adaptive Statistical Detection

- [x] Volatility robustness experiments
- [x] Historical volatility estimator
- [x] Adaptive CUSUM detector
- [x] Automated tests
- [x] Independent calibration and evaluation
- [x] Original versus adaptive CUSUM comparison

### Phase 3 — Adaptive Risk Controller

- [ ] Implement simulated exposure-reduction decisions
- [ ] Compare monitored and unmonitored strategies
- [ ] Model false-alarm and delayed-detection costs
- [ ] Evaluate drawdown, turnover, and risk-adjusted returns

### Phase 4 — Historical Validation

- [ ] Integrate historical market data
- [ ] Use chronological out-of-sample evaluation
- [ ] Model transaction costs and slippage
- [ ] Test robustness under realistic market conditions

### Phase 5 — Research Portfolio

- [ ] Produce publication-quality figures
- [ ] Build a research dashboard
- [ ] Prepare a formal technical report
- [ ] Document limitations and failure cases

## 11. Research Integrity

EDGEGUARD is an educational quantitative research project.

All reported experimental results are derived from synthetic simulations.

The project does not demonstrate profitable live trading, establish statistically significant superiority of adaptive CUSUM, or claim the invention of a novel change-detection algorithm.

Its longer-term objective is to evaluate whether carefully calibrated statistical monitoring can support more reliable simulated trading risk-control decisions.

## Phase 3 — Adaptive Risk Controller

### Overview

Phase 3 investigates whether statistical deterioration
signals can improve simulated trading risk management.

EDGEGUARD combines a volatility-normalized adaptive
CUSUM detector with an exposure-reduction controller.

When deterioration is detected, the controller reduces
strategy exposure beginning with the next observation,
avoiding same-period look-ahead bias.

### Implemented Components

- Adaptive risk controller (`risk_controller.py`)
- Transaction-cost model (`transaction_costs.py`)
- Commission and slippage sensitivity analysis
- Monte Carlo performance evaluation
- False-alarm economic cost analysis
- Exposure-policy comparison
- Robustness testing
- Bootstrap confidence intervals

### Experiments

| Experiment | Description |
|---|---|
| 14 | Risk-controller baseline evaluation |
| 15 | Adaptive CUSUM integration |
| 16 | Monte Carlo risk-controller evaluation |
| 17 | False-alarm economic cost |
| 18 | Exposure-reduction policy comparison |
| 19 | Transaction-cost sensitivity |
| 20 | Robustness across simulated market conditions |
| 21 | Bootstrap confidence intervals |

### Main Findings

**Experiment 16 — Monte Carlo Evaluation**

Across 500 synthetic deterioration paths:

- Average return improvement: +22.42 percentage points
- Average drawdown reduction: 14.49 percentage points
- Return improved on 89.0% of paths
- Drawdown reduced on 92.6% of paths

**Experiment 17 — False-Alarm Costs**

Across 500 no-change simulations:

- Average return difference: -1.78 percentage points
- Average drawdown reduction: 0.12 percentage points
- Average conditional return difference following
  false alarms: -42.33 percentage points

These results illustrate the opportunity cost of
unnecessary exposure reductions.

**Experiment 19 — Transaction Costs**

Three proportional-cost scenarios were evaluated:

- Low: 2 basis points
- Medium: 10 basis points
- High: 30 basis points

The modeled costs had relatively small effects because
the controller makes at most one exposure reduction.

**Experiment 20 — Robustness**

The controller was evaluated across 36 combinations
of volatility, deterioration magnitude, and change point,
using 3,600 simulated paths.

Detection reliability decreased substantially under
higher volatility and weaker deterioration.

Some scenarios produced negative average return
improvements.

**Experiment 21 — Statistical Confidence Intervals**

Across 500 independent synthetic deterioration paths
and 2,000 bootstrap resamples:

| Metric | Mean improvement | 95% bootstrap CI |
|---|---:|---|
| Total return | +22.79 pp | [+21.46, +24.19] pp |
| Maximum drawdown reduction | +14.63 pp | [+13.96, +15.30] pp |

These intervals quantify simulation uncertainty under
the specified model. They do not establish real-market
profitability or account for all model-selection uncertainty.

### Limitations

- Returns are generated using simplified Gaussian models.
- The simulations do not represent historical market data.
- Transaction costs use a simplified proportional model.
- Financing, nonlinear market impact, and portfolio-weight
  drift are not modeled.
- The exposure policy is not statistically proven optimal.
- Performance depends on the assumed deterioration regime.
- Real-world trading effectiveness remains unverified.

### Verification

The complete automated test suite passed 38 tests.

Python syntax checks completed successfully.

### Conclusion

Phase 3 demonstrates that adaptive exposure reduction
can improve simulated outcomes when trading strategy
profitability deteriorates.

However, unnecessary alarms can sacrifice profits,
and detection reliability declines under difficult
market conditions.

Further validation using historical data, realistic
execution assumptions, and independent out-of-sample
testing is necessary.
