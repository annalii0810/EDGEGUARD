
# EDGEGUARD: Evaluating Adaptive Change-Point Detection for Quantitative Trading Risk Management

**Research type:** Independent undergraduate quantitative research  
**Status:** Completed historical evaluation and exploratory synthetic experiments  
**Date:** October 2026

---

## Abstract

This study investigates whether adaptive change-point detection can identify deterioration in quantitative trading strategies and improve risk management through dynamic exposure reduction.

EDGEGUARD combines exposure-aware adaptive cumulative sum (CUSUM) monitoring, causal volatility estimation, and a controller that reduces trading exposure following detection.

The research evaluates historical exchange-traded fund data, controlled synthetic deterioration scenarios, and conventional statistical detector baselines.

In a frozen historical evaluation covering January 2023 through October 2026, EDGEGUARD generated no alarms across SPY, QQQ, IWM, and TLT. Consequently, the risk controller produced no measured improvement in returns, maximum drawdowns, or Sharpe ratios.

Under controlled synthetic deterioration, EDGEGUARD improved mean total returns by 27.09 percentage points and reduced mean maximum drawdown by 24.09 percentage points relative to an unprotected strategy.

However, conventional standardized CUSUM slightly outperformed EDGEGUARD in direct synthetic economic comparisons.

These findings demonstrate conditional benefits of change-point-based exposure control under simulated deterioration, but do not establish improved historical trading performance or general superiority over conventional detectors.

---

## 1. Introduction

Quantitative trading strategies may deteriorate when market conditions change, statistical relationships weaken, or previously profitable signals become ineffective.

A major challenge in systematic trading is distinguishing temporary losses from persistent deterioration.

Risk-management systems face two competing objectives:

1. Detect genuine deterioration quickly enough to reduce losses.
2. Avoid unnecessary alarms that sacrifice profitable trading opportunities.

An overly sensitive detector may frequently reduce exposure during normal conditions.

A conservative detector may respond too slowly to prevent substantial losses.

EDGEGUARD investigates whether adaptive, exposure-aware CUSUM monitoring can improve this tradeoff.

### Research Questions

1. Can adaptive CUSUM detect persistent deterioration in trading-strategy returns?
2. Does reducing exposure following an alarm improve simulated economic outcomes?
3. Does adaptive volatility estimation provide advantages over conventional fixed-volatility detection?
4. How sensitive is detection performance to deterioration severity and volatility?
5. Under which conditions does statistical monitoring provide meaningful economic protection?

---

## 2. Historical Data and Methodology

### 2.1 Historical Market Data

Historical adjusted closing prices were downloaded using Yahoo Finance through the `yfinance` Python package.

Four exchange-traded funds were examined:

- SPY — U.S. large-cap equities.
- QQQ — Nasdaq-100 equities.
- IWM — U.S. small-cap equities.
- TLT — Long-duration U.S. Treasury bonds.

The original dataset contains daily observations beginning January 2010 and ending October 27, 2026.

Daily simple returns were calculated as:

$$
R_t = \frac{P_t}{P_{t-1}} - 1
$$

where $P_t$ represents the adjusted closing price.

Adjusted-price returns do not incorporate all practical execution costs, market impact, or liquidity constraints.

### 2.2 Chronological Data Splitting

Historical observations were divided into:

| Period | Dates | Purpose |
|---|---|---|
| Training | 2010–2018 | Reference estimation and development |
| Validation | 2019–2022 | Methodology evaluation |
| Final test | 2023–October 2026 | Reserved out-of-sample evaluation |

The final test period was evaluated after the historical methodology and detector parameters were frozen.

### 2.3 Trading Strategy

A 50-day moving-average strategy provides the underlying trading exposure.

The moving average is:

$$
MA_t = \frac{1}{50}\sum_{i=t-49}^{t}P_i
$$

The trading signal is:

$$
S_t =
\begin{cases}
1, & P_t > MA_t \\
0, & P_t \leq MA_t
\end{cases}
$$

Exposure for observation $t$ is determined using the preceding signal:

$$
E_t = S_{t-1}
$$

This avoids using the current closing price to determine exposure for the return already realized during the current observation.

The strategy return is:

$$
R_t^{strategy} = E_t R_t
$$

### 2.4 Adaptive CUSUM Detection

EDGEGUARD uses a downward cumulative sum statistic to monitor negative deviations in active strategy returns.

The detector normalizes observed active returns using estimated historical volatility.

The conceptual adaptive CUSUM statistic is:

$$
A_t =
\max\left(
0,
A_{t-1}
+
\frac{\mu_0-X_t}
{\max(\hat{\sigma}_t,\sigma_{\min})}
-k
\right)
$$

where:

- $A_t$ is the adaptive detection statistic.
- $\mu_0$ is the reference expected return.
- $X_t$ is the normalized active return.
- $\hat{\sigma}_t$ is the causal volatility estimate.
- $\sigma_{\min}$ is the volatility floor.
- $k$ is the standardized allowance.

The historical detector uses a warm-up period and preserves its statistic during inactive observations.

### 2.5 Frozen Historical Parameters

| Parameter | Value |
|---|---:|
| Strategy window | 50 |
| Detector window | 50 |
| Reference mean | 0.0002299069244105812 |
| Allowance | 0.1 |
| Detection threshold | 25.0 |
| Minimum volatility | 0.001 |
| Reduced exposure multiplier | 0.25 |
| Transaction costs | 5 basis points |

These settings were preserved for the final historical evaluation.

### 2.6 Risk-Control Policy

After the first detector alarm, the controller reduces the underlying strategy exposure to 25% of its original level.

If an alarm occurs at observation $t$, the reduction begins at observation $t+1$.

The alarm observation itself retains its original exposure.

The policy remains reduced for the remainder of the evaluation period.

### 2.7 Transaction-Cost Accounting

Turnover is calculated as:

$$
T_t = |E_t-E_{t-1}|
$$

Transaction costs are:

$$
C_t = T_t\frac{b}{10000}
$$

where $b$ is the number of basis points.

Net returns are:

$$
R_t^{net} = E_tR_t-C_t
$$

The historical backtester charges five basis points per unit of turnover.

Initial exposure is supplied separately to account for position changes at evaluation boundaries.

---

## 3. Historical Out-of-Sample Evaluation

The frozen historical test period covered January 3, 2023, through October 27, 2026.

### 3.1 Final Results

| Asset | Baseline Return | EDGEGUARD Return | Alarm |
|---|---:|---:|---|
| SPY | +56.38% | +56.38% | None |
| QQQ | +83.96% | +83.96% | None |
| IWM | +30.32% | +30.32% | None |
| TLT | -21.64% | -21.64% | None |

No detector alarms occurred across the four assets.

Consequently, EDGEGUARD produced no measured improvement in total returns, maximum drawdowns, or Sharpe ratios.

### 3.2 Interpretation

The historical findings do not support a claim that EDGEGUARD improves trading performance under the frozen configuration.

The absence of alarms may reflect the detector threshold, the underlying trading strategy, the market conditions, or interactions among these factors.

The original historical results were preserved without retrospective parameter changes.

---

## 4. Controlled Synthetic Deterioration

### 4.1 Simulation Design

Synthetic returns were generated using:

$$
R_t = \mu_t+\sigma Z_t
$$

where:

$$
Z_t \sim \mathcal{N}(0,1)
$$

The expected return changes at a predetermined observation:

$$
\mu_t =
\begin{cases}
\mu_0, & t<\tau \\
\mu_1, & t\geq\tau
\end{cases}
$$

The reference Phase 5 scenario used:

| Parameter | Value |
|---|---:|
| Simulation paths | 500 |
| Observations per path | 500 |
| Change point | 250 |
| Reference mean | Approximately 0.00023 |
| Deteriorated mean | -0.003 |
| Volatility | 0.00715 |

### 4.2 Detection Performance

| Metric | Result |
|---|---:|
| False-alarm rate | 3.2% |
| Premature-alarm rate | 1.4% |
| Post-change detection rate | 98.6% |
| Average detection delay | Approximately 59.5 observations |

These results demonstrate detection capability under the specified synthetic assumptions.

### 4.3 Deterioration Sensitivity

Detection probability increased as the negative mean shift became stronger.

Small changes produced lower detection probabilities and longer delays.

Large persistent negative shifts were detected more frequently and more quickly.

### 4.4 Volatility Sensitivity

Detection reliability declined as volatility increased.

At simulated volatility of 0.02, detection probability fell to 59.6%.

This illustrates how return noise can obscure deterioration.

---

## 5. Conventional Detector Comparisons

EDGEGUARD was compared against rolling-mean detection and standardized CUSUM.

Detector thresholds were calibrated using synthetic no-change paths to target approximately 5% false-alarm probability over a specified monitoring horizon.

Independent synthetic paths were then used for evaluation.

### 5.1 Detection Within Fixed Deadlines

| Detector | Within 50 Observations | Within 100 Observations |
|---|---:|---:|
| EDGEGUARD | 39.0% | 93.1% |
| Rolling Mean | 59.0% | 88.2% |
| Standard CUSUM | 44.0% | 94.2% |

EDGEGUARD detected fewer deteriorations than Rolling Mean within 50 observations.

At 100 observations, EDGEGUARD exceeded Rolling Mean but remained slightly behind Standard CUSUM.

### 5.2 Paired Statistical Comparisons

Relative to Rolling Mean:

- EDGEGUARD's detection probability was 20.0 percentage points lower within 50 observations.
- EDGEGUARD's detection probability was 4.9 percentage points higher within 100 observations.

Among paths where both detectors detected deterioration, EDGEGUARD exhibited longer average detection delays.

These findings do not establish general statistical superiority.

---

## 6. Synthetic Economic Risk-Control Evaluation

### 6.1 No-Change Scenario

Under a simulated environment without deterioration:

| Metric | Baseline | EDGEGUARD |
|---|---:|---:|
| Mean total return | +11.46% | +11.21% |
| Mean maximum drawdown | 14.47% | 14.32% |
| Mean Sharpe ratio | 0.484 | 0.462 |

EDGEGUARD reduced mean return by 0.25 percentage points.

This demonstrates the opportunity cost of unnecessary exposure reductions.

### 6.2 Deterioration Scenario

Under severe persistent deterioration:

| Metric | Baseline | EDGEGUARD |
|---|---:|---:|
| Mean total return | -50.15% | -23.05% |
| Mean maximum drawdown | 56.19% | 32.10% |
| Mean Sharpe ratio | -3.006 | -1.432 |

EDGEGUARD improved mean total return by 27.09 percentage points and reduced mean maximum drawdown by 24.09 percentage points.

### 6.3 Bootstrap Confidence Intervals

Paired bootstrap confidence intervals were:

| Metric | Mean Improvement | 95% Bootstrap CI |
|---|---:|---|
| Total return | +27.09 pp | +26.70 to +27.49 pp |
| Drawdown reduction | +24.09 pp | +23.80 to +24.40 pp |

These intervals quantify simulation uncertainty under the specified assumptions.

They do not establish expected real-market performance.

---

## 7. Economic Comparison Against Standard CUSUM

A separate synthetic experiment compared EDGEGUARD against a conventional standardized CUSUM risk controller.

### 7.1 Deterioration Results

| Metric | EDGEGUARD | Standard CUSUM |
|---|---:|---:|
| Mean total return | -23.46% | -23.05% |
| Mean maximum drawdown | 32.10% | 31.73% |

EDGEGUARD's mean return disadvantage was approximately 0.42 percentage points.

The paired 95% bootstrap interval for the return difference was approximately:

$$
[-0.52,\,-0.30]
$$

in percentage points.

Standard CUSUM slightly outperformed EDGEGUARD under this scenario.

### 7.2 Interpretation

Adaptive volatility estimation did not produce a general economic advantage over the conventional baseline in the tested setting.

This result is important because improvement over an unprotected strategy does not necessarily demonstrate that a particular detector adds value relative to simpler alternatives.

---

## 8. Volatility Misspecification

Additional simulations examined how incorrect volatility assumptions affected conventional CUSUM.

| Assumed / True Volatility | Standard CUSUM False-Alarm Rate |
|---|---:|
| 0.50 | 85.8% |
| 0.75 | 28.8% |
| 1.00 | 5.6% |
| 1.25 | 0.8% |
| 1.50 | 0.0% |
| 2.00 | 0.0% |

EDGEGUARD produced a 4.8% false-alarm rate under the fixed simulated volatility.

These results demonstrate sensitivity to volatility misspecification in the conventional fixed-volatility detector.

However, they do not establish general robustness superiority over properly calibrated adaptive alternatives.

---

## 9. Software Validation and Reproducibility

EDGEGUARD includes:

- Chronological historical data splitting.
- Causal trading-signal construction.
- Exposure-aware statistical monitoring.
- Transaction-cost-aware backtesting.
- Detector warm-up and reset behavior.
- Reproducible Monte Carlo experiments.
- Dataset SHA-256 fingerprints.
- Frozen historical evaluation parameters.
- Git version-control checkpoints.
- Automated unit and integration tests.

At the completion of Phase 5, the project contained 99 passing automated tests.

These tests provide evidence of correct behavior for tested software properties but do not prove complete methodological correctness.

The original historical datasets were excluded from Git and identified using recorded fingerprints.

Future data downloads may differ due to revisions by the market-data provider.

---

## 10. Research Limitations

### 10.1 Synthetic Model Assumptions

Most controlled experiments use Gaussian returns and predetermined deterioration processes.

Actual markets exhibit heavy tails, volatility clustering, autocorrelation, changing correlations, and more complex regime dynamics.

### 10.2 Historical Null Results

The frozen historical evaluation produced no detector alarms.

Therefore, no historical improvement in trading performance was demonstrated.

### 10.3 Baseline Comparability

Some conventional detector baselines use known simulated volatility.

These information assumptions differ from those of adaptive volatility estimation.

In the synthetic detector comparisons, conventional Standard CUSUM was supplied with the true volatility parameter of the simulated return distribution, while EDGEGUARD estimated volatility causally from observed returns. Consequently, the comparison evaluates EDGEGUARD against an oracle-informed fixed-volatility benchmark rather than two detectors operating under identical information constraints. Although the detectors were evaluated on identical return paths, their volatility information differed. The results should therefore not be interpreted as demonstrating the superiority of one equally informed deployable detector over another.

### 10.4 Exploratory Experimentation

Phase 5 experiments were developed sequentially.

Their results should not be interpreted as independent confirmation of a newly optimized detector.

### 10.5 Economic Modeling

The economic simulations use simplified proportional transaction costs.

Market impact, financing, execution slippage, and liquidity constraints are not fully represented.

### 10.6 Practical Deployment

No live trading or independent prospective market evaluation was conducted.

The practical effectiveness of the system remains unverified.

---

## 11. Conclusion

EDGEGUARD demonstrates that sequential statistical monitoring combined with exposure reduction can limit losses under certain controlled synthetic deterioration scenarios.

However, the frozen historical evaluation produced no measurable improvement, and conventional standardized CUSUM slightly outperformed EDGEGUARD in direct synthetic economic comparisons.

The economic usefulness of change-point-based risk control depends on deterioration severity, volatility, false-alarm frequency, detection delay, and the cost of reducing exposure.

The primary contribution of EDGEGUARD is a reproducible experimental framework for investigating these relationships.

Future research should examine more realistic volatility processes, heavy-tailed returns, alternative adaptive detectors, recovery-aware risk policies, and independent prospective evaluation.

The present results do not establish a novel superior detector or profitable deployable trading strategy.

---

## 12. Research Integrity Statement

The original historical evaluation parameters were frozen before the final test-period evaluation.

Subsequent exploratory synthetic experiments did not alter the original historical configuration.

Positive, negative, and null findings are reported.

EDGEGUARD is presented as an independent quantitative research investigation rather than a validated profitable trading system.

---

## Research Documentation

- `README.md`
- `docs/evaluation_protocol.md`
- `docs/phase4_results.md`
- `docs/reproducibility.md`
- `research/phase5_findings.md`
- `research/phase5_plan.md`

The research report summarizes the implemented methodology and experimental findings. Further independent review and verification are required before academic submission.
