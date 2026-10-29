
# EDGEGUARD — Phase 5 Research Plan

## Objective

Investigate whether sequential deterioration detection
can provide statistically reliable improvements in
quantitative trading risk management.

## Research Questions

1. Can the detector identify genuine changes in
   strategy-return distributions?

2. How do false alarms and detection delays affect
   trading performance?

3. Under what market conditions does adaptive
   exposure reduction improve risk-adjusted returns?

## Research Workstreams

### A. Statistical Detection

- Simulate known deterioration events.
- Measure detection probability and delay.
- Estimate false-alarm rates with uncertainty.
- Compare against alternative change detectors.

### B. Risk-Control Evaluation

- Evaluate multiple predefined deterioration scenarios.
- Quantify transaction-cost sensitivity.
- Measure return, drawdown, and Sharpe differences.
- Compare against simple risk-management baselines.

### C. Robustness

- Use independent random seeds.
- Evaluate multiple volatility regimes.
- Report confidence intervals.
- Test sensitivity to modeling assumptions.

### D. Research Reproducibility

- Preserve the original Phase 4 results.
- Version all new experimental configurations.
- Record random seeds and software versions.
- Distinguish exploratory and confirmatory analyses.

## Publication Criteria

A research paper should require:

- A clearly defined research contribution.
- Appropriate statistical comparisons.
- Reproducible experiments.
- Transparent reporting of negative results.
- Evidence supporting conclusions.
- Explicit limitations.

## Research Integrity

The original 2023–2026 out-of-sample period
has already been evaluated.

It must not be reused as an untouched test set
for newly optimized models.
