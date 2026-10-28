
# EDGEGUARD — Phase 4 Final Results

## Evaluation

Experiment: 30
Period: January 3, 2023 – October 27, 2026

Frozen methodology commit: 12b293d
Final evaluation script commit: 401cc70

Assets: SPY, QQQ, IWM, TLT

## Results

| Asset | Baseline Return | EDGEGUARD Return | Max CUSUM | Alarms |
|---|---:|---:|---:|---:|
| SPY | 56.38% | 56.38% | 13.714 | 0 |
| QQQ | 83.96% | 83.96% | 11.525 | 0 |
| IWM | 30.32% | 30.32% | 9.070 | 0 |
| TLT | -21.64% | -21.64% | 11.105 | 0 |

## Findings

The frozen historical adaptive CUSUM detector
generated no alarms across the four assets.

EDGEGUARD therefore produced identical returns,
drawdowns, and Sharpe ratios to the baseline
moving-average strategy.

No out-of-sample performance improvement was
demonstrated under the frozen configuration.

## Interpretation

The detector threshold was not exceeded during
the reserved evaluation period.

These results do not establish that the risk
controller is effective at reducing historical
drawdowns.

The findings are consistent with a conservative
detector configuration, but do not independently
establish why alarms were absent.

## Research Integrity

The reserved test was evaluated after freezing
the detector configuration and evaluation script.

The test results must not be used to retune the
original model and claim another untouched
out-of-sample evaluation.

## Future Research

Future experiments may investigate alternative
detector formulations, calibration methods,
risk-control policies, and additional datasets.

Such investigations must be identified as new
research and evaluated on genuinely new data.
