
# EDGEGUARD

## Adaptive Change-Point Detection for Quantitative Trading Risk Management

**Independent Quantitative Research | Python | Statistics | Risk Management**

EDGEGUARD is a reproducible Python research project investigating whether sequential change-point detection can identify deterioration in quantitative trading strategies and improve risk management through dynamic exposure reduction.

The project combines adaptive CUSUM detection, causal volatility estimation, Monte Carlo simulation, historical backtesting, and statistical evaluation.

**Project Status:** Five research phases implemented, Experiments 01–41 completed, and 99 automated tests passing.

**Research Conclusion:** EDGEGUARD demonstrated substantial risk-reduction benefits under selected synthetic deterioration scenarios but produced no improvement in its frozen historical out-of-sample evaluation.

The project does not claim profitable live trading or general superiority over conventional statistical detectors.

---

## 1. Research Overview

Quantitative trading strategies may deteriorate when market conditions change or previously effective statistical relationships weaken.

A risk-monitoring system must balance two competing objectives:

1. Detect genuine deterioration quickly.
2. Avoid false alarms that unnecessarily reduce profitable trading exposure.

EDGEGUARD investigates the relationship between statistical detection, volatility estimation, detection delay, and economic risk management.

### Research Questions

- Can adaptive CUSUM detect persistent deterioration?
- Does reducing exposure after detection improve performance?
- How does EDGEGUARD compare against conventional detectors?
- How does volatility affect detection reliability?
- Under what conditions does risk reduction become economically beneficial?

### Technical Components

- Rolling-mean detection
- Conventional CUSUM
- Adaptive CUSUM
- Causal volatility estimation
- Exposure-aware risk monitoring
- Dynamic exposure reduction
- Transaction-cost-aware backtesting
- Monte Carlo simulations
- Threshold calibration
- Bootstrap confidence intervals
- Automated testing

---

## 2. Key Research Findings

### 2.1 Historical Out-of-Sample Evaluation

The frozen historical evaluation covered January 2023 through October 2026.

| Asset | Baseline Return | EDGEGUARD Return | Alarm |
|---|---:|---:|---|
| SPY | +56.38% | +56.38% | None |
| QQQ | +83.96% | +83.96% | None |
| IWM | +30.32% | +30.32% | None |
| TLT | -21.64% | -21.64% | None |

No detector alarms occurred.

Therefore, EDGEGUARD produced no measured improvement in returns, drawdowns, or Sharpe ratios.

### 2.2 Synthetic Deterioration Detection

Under severe, persistent simulated deterioration:

| Metric | Result |
|---|---:|
| Post-change detection rate | 98.6% |
| False-alarm rate | 3.2% |
| Premature-alarm rate | 1.4% |
| Average detection delay | Approximately 59.6 observations |

These results depend on the specified synthetic assumptions.

### 2.3 Synthetic Risk-Control Performance

| Metric | Baseline | EDGEGUARD |
|---|---:|---:|
| Mean total return | -50.15% | -23.05% |
| Mean maximum drawdown | 56.19% | 32.10% |
| Mean Sharpe ratio | -3.006 | -1.432 |

EDGEGUARD achieved:

- Mean return improvement: +27.09 percentage points.
- Mean drawdown reduction: 24.09 percentage points.

Paired bootstrap confidence intervals:

| Metric | 95% Confidence Interval |
|---|---|
| Return improvement | +26.70 to +27.49 pp |
| Drawdown reduction | +23.80 to +24.40 pp |

These results describe controlled simulations, not real-market profitability.

### 2.4 Conventional Detector Comparison

| Detector | Detection Within 50 Observations | Within 100 Observations |
|---|---:|---:|
| EDGEGUARD | 39.0% | 93.1% |
| Rolling Mean | 59.0% | 88.2% |
| Standard CUSUM | 44.0% | 94.2% |

EDGEGUARD did not consistently outperform conventional detection methods.

In a separate synthetic economic comparison, Standard CUSUM slightly outperformed EDGEGUARD.

### 2.5 Volatility Misspecification

| Assumed / True Volatility | Standard CUSUM False-Alarm Rate |
|---|---:|
| 0.50 | 85.8% |
| 0.75 | 28.8% |
| 1.00 | 5.6% |
| 1.25 | 0.8% |
| 1.50 | 0.0% |
| 2.00 | 0.0% |

EDGEGUARD produced a 4.8% false-alarm rate under the fixed simulated volatility.

This demonstrates sensitivity to volatility misspecification in the fixed-volatility baseline, not general superiority of EDGEGUARD.

---

## 3. Mathematical Framework

### 3.1 Synthetic Return Model

Synthetic returns follow:

$$
R_t = \mu_t + \sigma Z_t
$$

where:

- $R_t$ is the simulated return.
- $\mu_t$ is expected return.
- $\sigma$ is volatility.
- $Z_t$ is standard normal noise.

$$
Z_t \sim \mathcal{N}(0,1)
$$

### 3.2 Regime Deterioration

Expected return changes at observation $\tau$:

$$
\mu_t =
\begin{cases}
\mu_0, & t < \tau \\
\mu_1, & t \geq \tau
\end{cases}
$$

where:

- $\mu_0$ is the original expected return.
- $\mu_1$ is the deteriorated expected return.
- $\tau$ is the change point.

Phase 5 reference simulation:

| Parameter | Value |
|---|---:|
| Observations | 500 |
| Change point | 250 |
| Reference mean | 0.0002299069244105812 |
| Deteriorated mean | -0.003 |
| Volatility | 0.00715 |

### 3.3 Rolling-Mean Detector

The rolling mean is:

$$
\bar{R}_t =
\frac{1}{w}
\sum_{i=t-w+1}^{t}R_i
$$

The standardized detection statistic is:

$$
M_t =
\max\left(
0,
\frac{
\sqrt{w}(\mu_0-\bar{R}_t)
}{
\sigma_{\mathrm{ref}}
}
\right)
$$

An alarm occurs when:

$$
M_t > h_M
$$

The current observation contributes to the statistic.

Therefore, any resulting trading action must occur no earlier than the next observation.

### 3.4 Conventional CUSUM

The downward CUSUM statistic is:

$$
S_t =
\max\left(
0,
S_{t-1}+\mu_0-R_t-\delta
\right)
$$

where:

- $S_t$ is the accumulated statistic.
- $\mu_0$ is the reference mean.
- $\delta$ is the allowance.

An alarm occurs when:

$$
S_t > h_S
$$

### 3.5 Standardized CUSUM

The standardized CUSUM baseline uses:

$$
C_t =
\max\left(
0,
C_{t-1}
+
\frac{\mu_0-R_t}{\sigma_{\mathrm{ref}}}
-k
\right)
$$

where:

- $\sigma_{\mathrm{ref}}$ is fixed reference volatility.
- $k$ is the standardized allowance.

An alarm occurs when:

$$
C_t > h_C
$$

### 3.6 Causal Volatility Estimation

The rolling historical mean is:

$$
\bar{R}_{t-1}^{(w)}
=
\frac{1}{w}
\sum_{i=t-w}^{t-1}R_i
$$

The sample volatility estimate is:

$$
\hat{\sigma}_t
=
\sqrt{
\frac{1}{w-1}
\sum_{i=t-w}^{t-1}
\left(
R_i-\bar{R}_{t-1}^{(w)}
\right)^2
}
$$

Only observations before time $t$ contribute to the estimate.

### 3.7 Exposure-Aware Adaptive CUSUM

Let the original trading strategy exposure be:

$$
E_t^{(0)} \in [0,1]
$$

The strategy return is:

$$
R_t^{\mathrm{strategy}}
=
E_t^{(0)}R_t^{\mathrm{asset}}
$$

During active exposure, the detector uses:

$$
X_t =
\frac{
R_t^{\mathrm{strategy}}
}{
E_t^{(0)}
},
\qquad E_t^{(0)}>0
$$

Let $H_t$ contain the indices of the most recent $w$ active observations strictly before $t$.

The historical active-return mean is:

$$
\bar{X}_{H_t}
=
\frac{1}{w}
\sum_{i\in H_t}X_i
$$

The active-return volatility estimate is:

$$
\hat{\sigma}_t^{\mathrm{active}}
=
\sqrt{
\frac{1}{w-1}
\sum_{i\in H_t}
(X_i-\bar{X}_{H_t})^2
}
$$

After the active-return warm-up, EDGEGUARD updates:

$$
A_t =
\max\left(
0,
A_{t-1}
+
\frac{
\mu_0-X_t
}{
\max(
\hat{\sigma}_t^{\mathrm{active}},
\sigma_{\min}
)
}
-k
\right)
$$

where:

- $A_t$ is the adaptive statistic.
- $\mu_0$ is the reference mean.
- $\sigma_{\min}$ is the volatility floor.
- $k$ is the allowance.

During inactive observations:

$$
A_t=A_{t-1}
$$

An alarm occurs when:

$$
A_t>h_A
$$

The historical detector records the first alarm while continuing to calculate the statistic.

### 3.8 Dynamic Exposure Control

Let $t_a$ be the first alarm observation.

The exposure multiplier is:

$$
m_t =
\begin{cases}
1, & t\leq t_a \\
0.25, & t>t_a
\end{cases}
$$

The controlled exposure is:

$$
E_t^{\mathrm{controlled}}
=
E_t^{(0)}m_t
$$

If no alarm occurs:

$$
m_t=1
$$

Risk reduction begins on the observation following the alarm.

### 3.9 Transaction Costs

Exposure turnover is:

$$
T_t =
|E_t-E_{t-1}|
$$

Transaction cost is:

$$
C_t =
T_t\frac{b}{10000}
$$

where $b$ represents basis points.

Gross return is:

$$
R_t^{\mathrm{gross}}
=
E_tR_t^{\mathrm{asset}}
$$

Net return is:

$$
R_t^{\mathrm{net}}
=
E_tR_t^{\mathrm{asset}}-C_t
$$

The historical backtest uses:

$$
b=5
$$

### 3.10 Equity and Maximum Drawdown

Starting equity is:

$$
V_0=1
$$

The compounded equity curve is:

$$
V_t =
\prod_{i=1}^{t}
(1+R_i^{\mathrm{net}})
$$

The running peak is:

$$
P_t =
\max(V_0,V_1,\ldots,V_t)
$$

The Python implementation stores drawdown as:

$$
DD_t =
\frac{V_t}{P_t}-1
$$

Maximum drawdown is reported as a positive magnitude:

$$
\mathrm{MDD}
=
-\min_t DD_t
$$

### 3.11 Annualized Sharpe Ratio

The backtester calculates:

$$
\widehat{SR}
=
\sqrt{252}
\frac{
\overline{R}^{\mathrm{net}}
}{
s(R^{\mathrm{net}})
}
$$

where:

- $\overline{R}^{\mathrm{net}}$ is average daily net return.
- $s(R^{\mathrm{net}})$ is sample standard deviation.
- 252 is the assumed number of trading days per year.

The calculation assumes a zero risk-free rate.

---

## 4. Statistical Evaluation

### 4.1 Threshold Calibration

Thresholds are estimated using no-change simulations.

For target false-alarm probability $\alpha$:

$$
h =
Q_{1-\alpha}(M_1,\ldots,M_n)
$$

where $M_i$ is the maximum detector statistic from simulation $i$.

### 4.2 Alarm Classification

- **Premature:** First alarm before the change point.
- **Successful:** First alarm at or after the change point.
- **Missed:** No alarm before the observation horizon ends.

### 4.3 Detection Delay

$$
D =
t_{\mathrm{alarm}}-\tau
$$

Mean detection delay is calculated among successfully detected paths.

### 4.4 Paired Statistical Comparison

For paired outcome differences:

$$
d_i =
Y_{i,\mathrm{EDGEGUARD}}
-
Y_{i,\mathrm{baseline}}
$$

The sample mean difference is:

$$
\bar{d}
=
\frac{1}{n}
\sum_{i=1}^{n}d_i
$$

Paired bootstrap resampling is used to construct confidence intervals.

---

## 5. Historical Market Evaluation

### 5.1 Data

Historical adjusted closing prices were obtained using Yahoo Finance through `yfinance`.

Assets:

- SPY
- QQQ
- IWM
- TLT

Historical coverage: January 2010 through October 27, 2026.

### 5.2 Chronological Splits

| Period | Dates |
|---|---|
| Training | 2010–2018 |
| Validation | 2019–2022 |
| Final test | 2023–October 2026 |

### 5.3 Trading Strategy

The underlying strategy uses a 50-day moving average.

The previous closing signal determines the current exposure.

### 5.4 Frozen Detector Parameters

| Parameter | Value |
|---|---:|
| Strategy window | 50 |
| Detector window | 50 |
| Target mean | 0.0002299069244105812 |
| Allowance | 0.1 |
| Threshold | 25.0 |
| Minimum volatility | 0.001 |
| Reduced exposure | 0.25 |
| Transaction costs | 5 basis points |

### 5.5 Historical Findings

No alarms occurred across the four final test assets.

The controlled strategies therefore matched their unprotected baselines.

No historical improvement was demonstrated.

---

## 6. Research Phases

### Phase 1 — Statistical Foundations

**Experiments 01–07**

- Synthetic return simulation
- Rolling-mean detector
- Conventional CUSUM
- Threshold calibration
- Initial detector benchmarking

### Phase 2 — Adaptive Detection

**Experiments 08–13**

- Causal volatility estimation
- Adaptive CUSUM
- Volatility sensitivity
- Conventional detector comparisons

### Phase 3 — Risk Controller

**Experiments 14–21**

- Exposure reduction
- Transaction-cost modeling
- False-alarm opportunity costs
- Simulated drawdown analysis
- Bootstrap uncertainty estimates

### Phase 4 — Historical Validation

**Experiments 22–30**

- Historical data integration
- Chronological splitting
- Causal moving-average strategy
- Exposure-aware monitoring
- Frozen out-of-sample evaluation
- Documentation of null results

### Phase 5 — Advanced Research

**Experiments 31–41**

- Controlled deterioration simulations
- Mean-shift sensitivity
- Volatility regime sensitivity
- Conventional detector calibration
- Paired detector comparisons
- Economic risk-control simulations
- Volatility misspecification analysis
- Statistical confidence intervals

---

## 7. Repository Structure

    EDGEGUARD/
    ├── docs/
    │   ├── evaluation_protocol.md
    │   ├── phase4_results.md
    │   └── reproducibility.md
    ├── experiments/
    │   ├── experiment_01.py
    │   ├── ...
    │   └── experiment_41.py
    ├── research/
    │   ├── final_report.md
    │   ├── phase5_findings.md
    │   └── phase5_plan.md
    ├── scripts/
    │   └── download_market_data.py
    ├── src/
    │   └── edgeguard/
    │       ├── backtest.py
    │       ├── baseline_detectors.py
    │       ├── data_split.py
    │       ├── detectors.py
    │       ├── historical_data.py
    │       ├── historical_detector.py
    │       ├── historical_strategy.py
    │       ├── risk_controller.py
    │       ├── simulator.py
    │       ├── statistical_evaluation.py
    │       ├── transaction_costs.py
    │       └── volatility.py
    ├── tests/
    ├── .gitignore
    ├── README.md
    └── requirements.txt

---

## 8. Installation and Reproducibility

### Clone the Repository

    git clone https://github.com/annalii0810/EDGEGUARD.git
    cd EDGEGUARD

### Create a Virtual Environment

    python3 -m venv .venv
    source .venv/bin/activate

### Install Dependencies

    python -m pip install -r requirements.txt

### Run Automated Tests

    PYTHONPATH=src python -m pytest tests/ -q

The latest complete test suite passed 99 tests.

### Run Synthetic Experiments

    PYTHONPATH=src python experiments/experiment_35.py

    PYTHONPATH=src python experiments/experiment_36.py

    PYTHONPATH=src python experiments/experiment_37.py

    PYTHONPATH=src python experiments/experiment_39.py

    PYTHONPATH=src python experiments/experiment_40.py

    PYTHONPATH=src python experiments/experiment_41.py

### Historical Data Reproduction

The historical download script is:

    scripts/download_market_data.py

Historical datasets are excluded from Git.

See `docs/reproducibility.md` for dataset fingerprints and software versions.

The original historical test period has already been evaluated and must not be reused as an untouched test set.

---

## 9. Research Limitations

### Statistical

- Synthetic returns use simplified Gaussian assumptions.
- Deterioration follows predetermined mean shifts.
- Heavy tails and volatility clustering require further evaluation.
- Detector thresholds are not proven optimal.
- Detection delay estimates are conditional on successful detections.
- Multiple exploratory experiments introduce model-selection uncertainty.

### Historical

- No detector alarms occurred in the frozen final test.
- No historical economic improvement was demonstrated.
- Only four ETFs were evaluated.
- The underlying strategy is a simple moving-average benchmark.

### Economic

- Transaction costs use a simplified proportional model.
- Market impact and financing costs are not fully modeled.
- False alarms can sacrifice profitable exposure.
- Synthetic results do not establish live profitability.

### Detector Comparisons

- Some conventional baselines use true simulated volatility.
- Fixed-volatility methods are sensitive to scale misspecification.
- General superiority of EDGEGUARD has not been established.

---

## 10. Research Roadmap

### Phase 1 — Statistical Foundations

- [x] Return simulation
- [x] Rolling-mean detection
- [x] Conventional CUSUM
- [x] Threshold calibration

### Phase 2 — Adaptive Detection

- [x] Historical volatility estimation
- [x] Adaptive CUSUM
- [x] Robustness testing
- [x] Baseline comparisons

### Phase 3 — Risk Controller

- [x] Exposure reduction
- [x] Transaction-cost modeling
- [x] Economic evaluation
- [x] Bootstrap analysis

### Phase 4 — Historical Validation

- [x] Historical market data
- [x] Chronological splitting
- [x] Frozen detector configuration
- [x] Out-of-sample evaluation
- [x] Null-result documentation

### Phase 5 — Advanced Research

- [x] Controlled deterioration experiments
- [x] Volatility sensitivity
- [x] Paired detector comparisons
- [x] Economic simulations
- [x] Confidence intervals
- [x] Research report draft
- [ ] Final methodological review
- [ ] Independent academic review
- [ ] Final portfolio presentation

---

## 11. Research Integrity

EDGEGUARD is an independent quantitative research and software engineering project.

The frozen historical evaluation and later exploratory synthetic experiments are reported separately.

Positive, negative, and null findings are preserved.

The project does not establish profitable live trading, general statistical superiority, or a fundamentally novel detection algorithm.

The primary contribution is a reproducible framework for investigating statistical deterioration detection and economic risk-control decisions.

---

## 12. Research Documentation

- [Final Research Report](research/final_report.md)
- [Phase 5 Findings](research/phase5_findings.md)
- [Phase 5 Plan](research/phase5_plan.md)
- [Historical Evaluation Protocol](docs/evaluation_protocol.md)
- [Historical Results](docs/phase4_results.md)
- [Reproducibility Record](docs/reproducibility.md)

---

## 13. Future Research

Potential extensions include:

- GARCH volatility models
- Heavy-tailed return simulations
- Autocorrelated return processes
- Multiple deterioration and recovery regimes
- Alternative adaptive change-point detectors
- Recovery-aware exposure policies
- More realistic transaction-cost models
- Prospective evaluation on new market data

---

## Disclaimer

EDGEGUARD is an educational and exploratory quantitative research project.

The results are not investment advice or evidence of a profitable deployable trading strategy.
