
EDGEGUARD

Sequential Statistical Monitoring for Quantitative Trading Strategies

A reproducible Python research project investigating how sequential change-detection methods identify deterioration in simulated trading strategy profitability.
1. Project Overview

Quantitative trading strategies can lose their statistical advantage as market conditions change. Temporary losses do not necessarily mean the strategy has deteriorated, while a deteriorating strategy may still occasionally produce positive returns.

EDGEGUARD investigates the tradeoffs among false alarms, successful detection, missed changes, and detection delay. It implements rolling-mean monitoring, downward CUSUM, historical volatility estimation, and volatility-normalized adaptive CUSUM, then evaluates them on seeded synthetic returns.

Current implementation: Experiments 01–13 completed; 25 automated tests passed in the latest full test run. Phase 2 implementation and baseline comparisons are complete, while broader robustness validation remains future work.
2. Research Question

Can sequential statistical change-detection methods identify deterioration in a quantitative trading strategy's expected return while controlling false alarms and minimizing detection delay?

A related question is whether estimating volatility causally from historical returns produces performance comparable to an oracle CUSUM that knows the simulator's true volatility.
3. Mathematical Framework
3.1 Synthetic return model

The simulation generates independent Gaussian returns:
Rt=μt+σZt,Zt∼N(0,1).
Rt​=μt​+σZt​,Zt​∼N(0,1).

Here, RtRt​ is the observed return, μtμt​ is its expected value, and σσ is the standard deviation. The baseline regime-change scenario uses
μt={0.001t<500,−0.001t≥500.
μt​={0.001−0.001​t<500,t≥500.​

The simulated horizon is 1,000 observations. Unless otherwise specified, the change occurs at index 500, with baseline σ=0.01σ=0.01. Additional experiments vary σσ among 0.005, 0.010, 0.020, and 0.030. These are synthetic assumptions, not a model validated on live market returns.
3.2 Rolling-mean detector

For a window of ww observations:
R‾t=1w∑i=t−w+1tRi.
Rt​=w1​i=t−w+1∑t​Ri​.

The initial rule triggers when R‾t<0Rt​<0. A more conservative experimental rule triggers when
R‾t<−kσw.
Rt​<−kw
​σ​.

The kk-based rule is a heuristic for a sequential monitoring process; it does not itself guarantee a specified experiment-wide false-alarm probability.
3.3 Downward CUSUM

The original CUSUM accumulates negative deviations from a target return:
St=max⁡(0,St−1+μ0−Rt−δ),S−1=0.
St​=max(0,St−1​+μ0​−Rt​−δ),S−1​=0.

It triggers the first alarm when St>hSt​>h. The allowance δδ and threshold hh are measured in the raw-return scale.
3.4 Causal historical volatility estimator

For t≥wt≥w, the historical volatility estimator uses only the previous ww observations:
σ^t=std⁡sample(Rt−w,…,Rt−1).
σ
t​=stdsample​(Rt−w​,…,Rt−1​).

The implementation uses sample standard deviation (ddof=1), with NaN for the initial indices lacking a complete historical window. This construction excludes the current return and future observations from the volatility estimate.
3.5 Adaptive CUSUM

Adaptive CUSUM normalizes each deviation using the historical estimate and a minimum volatility floor:
At=max⁡(0,At−1+μ0−Rtmax⁡(σ^t,σmin⁡)−k).
At​=max(0,At−1​+max(σ
t​,σmin​)μ0​−Rt​​−k).

An alarm occurs when At>hAAt​>hA​. The statistic starts once a historical volatility window is available. The floor prevents division by zero but can magnify small deviations if set too low. Experiments therefore examine the floor's effect rather than treating it as automatically safe.

CUSUM, rolling means, and volatility normalization build on established statistical ideas. The current project does not claim a novel algorithm.
4. Experimental Methodology

    Reproducibility: NumPy-seeded synthetic simulations. Paired detector comparisons reuse the same simulated return sequences; these paired results must not be treated as independent replications.
    Premature alarm: First alarm occurs before the known change point ττ.
    Successful detection: First alarm occurs at or after ττ within the observation horizon.
    Missed detection: No alarm occurs within the horizon.
    Conditional detection delay: D=talarm−τD=talarm​−τ, averaged only across successful detections.
    No-change false-alarm proportion: Number of no-change simulations with at least one alarm divided by the total number of no-change simulations.

The first-alarm outcomes are mutually exclusive. Conditional delay must be interpreted alongside the success and miss rates. The false-alarm target is a probability of at least one alarm over 1,000 observations, not a per-observation probability.

Thresholds in the later experiments are selected using calibration simulations, then evaluated on separate no-change and regime-change simulations. Empirical 95th-percentile thresholds target approximately 5% no-change false alarms, subject to Monte Carlo uncertainty.
5. Experimental Results
Experiment 01 — Synthetic strategy edge deterioration

A 1,000-observation series changes expected return at index 500. In the recorded seeded run, the observed average before the change was +0.0869%, and after the change was −0.1447%; realized averages differ from theoretical means because of sampling noise.

Experiment 02 — Initial rolling-mean detection

With window 50, zero threshold, and 500 regime-change simulations: 498 premature alarms, 2 successful detections, and 0 missed detections. The premature-alarm proportion was 99.6%.
Experiment 03 — Rolling-window comparison
Window	Premature alarms	Successful detections	Conditional average delay
10	100.0%	0.0%	N/A
50	99.6%	0.4%	50.00
100	89.6%	10.4%	50.15
200	42.2%	57.8%	97.76
Experiment 04 — No-change false alarms
Window	Probability of at least one false alarm
10	100.0%
50	100.0%
100	98.6%
200	75.6%
Experiment 05 — Rolling-mean threshold comparison
Threshold	Premature alarms	Successful detections	Missed detections	Conditional delay
k=0k=0	99.6%	0.4%	0.0%	50.00
k=1k=1	78.8%	21.2%	0.0%	61.08
k=2k=2	16.4%	80.4%	3.2%	159.43
k=3k=3	0.8%	47.2%	52.0%	248.56

More conservative thresholds reduced premature alarms but could increase detection delay and missed detections.
Experiment 06 — Calibrated rolling-mean detection

A threshold of −0.003852 was selected using 1,000 no-change calibration simulations. Independent evaluation yielded 6.6% no-change false alarms, 3.8% premature alarms, 59.6% successful detections, 36.6% missed detections, and 213.17 conditional average delay.
Experiment 07 — Calibrated original CUSUM

A threshold of 0.273950 was selected at baseline volatility σ=0.01σ=0.01. Independent evaluation yielded 6.8% no-change false alarms, 2.8% premature alarms, 93.8% successful detections, 3.4% missed detections, and 195.42 conditional average delay.
Experiment 08 — Volatility robustness with fixed thresholds

The rolling-mean and original CUSUM detectors were tested across four volatility levels. Their fixed thresholds were not recalibrated for each volatility. The original CUSUM results were:
Volatility	Premature alarms	Successful detections	Missed detections	Conditional delay
0.005	0.0%	98.4%	1.6%	252.57
0.010	1.8%	94.4%	3.8%	191.29
0.020	73.8%	26.0%	0.2%	106.14
0.030	97.6%	2.4%	0.0%	55.25

The rising premature-alarm rate highlights the limitations of fixed, unnormalized thresholds when volatility changes.
Experiment 09 — Volatility-normalized CUSUM baseline

Using known simulated volatility for normalization, this exploratory baseline produced 12.6% premature alarms at each volatility level, with successful detection rates of 87.4%, 87.2%, 61.0%, and 40.2% as volatility increased. These thresholds were not yet independently calibrated to a common false-alarm target, so the results should not be used as a definitive detector ranking.
Experiment 10 — Calibrated normalized CUSUM with known volatility

The calibrated normalized threshold was 27.394951. Independent no-change false-alarm proportions were 6.8% at each tested volatility. Regime-change outcomes were:
Volatility	Premature alarms	Successful detections	Missed detections	Conditional delay
0.005	2.8%	97.2%	0.0%	77.80
0.010	2.8%	93.8%	3.4%	195.42
0.020	2.8%	41.8%	55.4%	262.39
0.030	2.8%	24.6%	72.6%	267.41

This is an oracle benchmark because it uses true simulator volatility, unavailable to a real deployment.
Experiment 11 — Historical volatility estimation

A causal rolling sample-standard-deviation estimator was implemented and tested. For [0.01, 0.02, 0.03, 0.04, 0.05] with window 3, the estimated series is [NaN, NaN, NaN, 0.01, 0.01]. Tests cover numeric values, no future-data dependence, invalid inputs, and zero-variance windows.
Experiment 12 — Adaptive CUSUM floor and threshold calibration

Three volatility floors (0.000001, 0.0005, and 0.001) were separately calibrated using 1,000 no-change simulations. Each yielded threshold 28.233946 at the displayed precision. On 500 independent no-change sequences per volatility, each candidate produced 31/500 = 6.2% false alarms at all four volatility levels.

All three floors produced the same reported regime-change results under these Gaussian settings:
Volatility	Premature alarms	Successful detections	Missed detections	Conditional delay
0.005	2.0%	98.0%	0.0%	78.75
0.010	2.0%	94.6%	3.4%	193.98
0.020	2.0%	43.6%	54.4%	267.10
0.030	2.0%	24.0%	74.0%	263.78

The floor candidates were indistinguishable in these tests; this does not establish equivalence under near-zero or time-varying volatility.
Experiment 13 — Controlled oracle-versus-adaptive CUSUM comparison

Both detectors were calibrated using 1,000 no-change sequences separately for each volatility, then evaluated on the same independent seeded sequences. The original oracle used raw-return CUSUM with allowance 0.1σ0.1σ; adaptive CUSUM used historical volatility (window 50), allowance 0.1, and minimum volatility 0.001.
Volatility	Oracle threshold	Adaptive threshold
0.005	0.136975	28.233946
0.010	0.273950	28.233946
0.020	0.547899	28.233946
0.030	0.821849	28.233946

Across 500 independent no-change simulations at each volatility, the oracle had 6.8% false alarms and adaptive CUSUM 6.2%. Results at different volatilities reuse the same random seeds and are not independent confirmations.
Volatility	Oracle success	Adaptive success	Oracle delay	Adaptive delay
0.005	97.2%	98.0%	77.80	78.75
0.010	93.8%	94.6%	195.42	193.98
0.020	41.8%	43.6%	262.39	267.10
0.030	24.6%	24.0%	267.41	263.78

The oracle had 2.8% premature alarms at each volatility; adaptive CUSUM had 2.0%. The adaptive detector matched the oracle broadly under these specific synthetic conditions, with small mixed differences. No confidence intervals or paired significance tests have yet established a statistically significant improvement. Conditional delays compare different subsets of detected simulations and should not be interpreted alone.
6. Key Findings

    The uncalibrated rolling-mean detector generated excessive false alarms in these simulations.
    Calibrating thresholds using independent no-change sequences greatly improved false-alarm control.
    CUSUM substantially outperformed the calibrated rolling-mean baseline in the baseline scenario, but this does not establish universal superiority.
    Fixed raw-return thresholds became unreliable when volatility increased.
    Historical volatility normalization made the adaptive detector's calibrated threshold largely invariant across the tested Gaussian scales.
    Adaptive CUSUM achieved broadly comparable simulated detection performance to a true-volatility oracle; superiority is not established.
    Both approaches struggled to detect the same small mean deterioration as volatility increased.

7. Repository Structure
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
│   ├── ...
│   └── experiment_13.py
├── results/
│   └── experiment_01.png
├── requirements.txt
├── .gitignore
└── README.md
8. Installation and Reproducibility

Requires Python 3, NumPy, Matplotlib, and pytest. The latest local test run used Python 3.14.7.
git clone https://github.com/annalii0810/EDGEGUARD.git
cd EDGEGUARD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

Run all automated tests:
PYTHONPATH=src python -m pytest -v

Latest observed local result: 25 passed. This documents the local test run; rerun the command to verify any new checkout.

Run the research experiments:
PYTHONPATH=src python experiments/experiment_01.py
PYTHONPATH=src python experiments/experiment_07.py
PYTHONPATH=src python experiments/experiment_08.py
PYTHONPATH=src python experiments/experiment_09.py
PYTHONPATH=src python experiments/experiment_10.py
PYTHONPATH=src python experiments/experiment_11.py
PYTHONPATH=src python experiments/experiment_12.py
PYTHONPATH=src python experiments/experiment_13.py

Experiments 02–06 can be run analogously. Some calibration and evaluation scripts run thousands of simulations and take longer than unit tests. All results reported above were obtained from seeded synthetic simulations.
9. Limitations and Research Integrity

    Returns are independent Gaussian draws, not realistic market microstructure or heavy-tailed, autocorrelated financial returns.
    Several experiments use true simulator parameters; those configurations are oracle benchmarks, not deployable systems.
    Per-volatility calibration in Experiment 13 assumes a known test volatility level when selecting the oracle threshold and adaptive threshold. This should not be confused with one globally deployable adaptive policy.
    The observed false-alarm percentages are finite-simulation estimates, not guaranteed levels.
    Different volatility conditions reuse random seeds; they are paired comparisons, not independent replications.
    The study lacks formal confidence intervals, paired statistical tests, sensitivity analyses over multiple independent evaluation sets, and adversarial distribution-shift experiments.
    The minimum volatility floor may amplify tiny deviations during near-constant returns; its safety has not been established across realistic regimes.
    Detection delays are conditional on successful detections and may be subject to selection effects.
    Historical market validation, transaction costs, slippage, execution latency, and real portfolio impact remain untested.
    Results do not establish live profitability, production readiness, or a novel statistical method.

10. Research Roadmap
Phase 1 — Statistical foundations

    Synthetic return generator and reproducibility
    Rolling-mean and original CUSUM baselines
    Initial unit tests and threshold calibration
    Initial synthetic benchmarking

Phase 2 — Adaptive statistical detection (implementation and initial evaluation)

    Evaluate several fixed-volatility scenarios
    Implement volatility-normalized CUSUM
    Implement causal historical volatility estimation
    Implement adaptive CUSUM and numerical safeguards
    Calibrate thresholds on separate no-change sequences
    Compare adaptive and oracle detectors on independent synthetic sequences
    Run 25 automated tests
    Extend to gradual changes, autocorrelation, heavy tails, and time-varying volatility
    Add confidence intervals, paired statistical tests, and repeated independent evaluations
    Investigate low-volatility safeguards under broader validation scenarios

Phase 3 — Adaptive risk controller

    Translate alarms into exposure-reduction and pause decisions
    Quantify costs of false alarms, delayed detection, and missed deterioration
    Compare risk-control policies using controlled simulations
    Evaluate risk-adjusted outcomes and policy stability

Phase 4 — Historical validation

    Integrate historical return data
    Use chronological out-of-sample evaluation
    Model transaction costs and slippage
    Study regime shifts and distributional instability

Phase 5 — Research portfolio

    Produce publication-quality figures and a technical report
    Build an interactive research dashboard
    Document failure cases and reproducibility details
    Prepare evidence-based resume descriptions

11. Research Integrity Statement

EDGEGUARD is an educational quantitative research project. Its currently reported findings arise from controlled synthetic simulations. The work investigates established sequential monitoring methods and does not claim profitable live trading, statistically proven superiority, or a novel algorithm. Its longer-term objective is to evaluate whether carefully calibrated monitoring can support more reliable trading risk-control decisions.