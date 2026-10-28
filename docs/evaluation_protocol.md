
# EDGEGUARD — Historical Evaluation Protocol

## Research Objective

Evaluate whether a calibrated, exposure-aware CUSUM
risk controller improves a moving-average trading
strategy's risk-adjusted performance.

## Historical Data

Assets: SPY, QQQ, IWM, TLT

Training: 2010–2018
Validation: 2019–2022
Reserved test: 2023–2026

The final test period is excluded from model selection.

## Trading Strategy

- Moving-average window: 50 observations
- Previous closing signal determines current exposure
- Long or cash positions
- Transaction costs: 5 basis points per unit turnover

## Frozen Detector Configuration

- Target mean: 0.0002299069244105812
- CUSUM allowance: 0.1
- Alarm threshold: 25.0
- Minimum volatility: 0.001
- Active-return window: 50 observations
- Exposure after alarm: 25% of strategy exposure

The target mean and threshold were calibrated
using SPY training data and synthetic simulations.

## Detector Policy

- Monitor original strategy returns.
- Use only previous active observations for volatility.
- Inactive periods preserve the CUSUM statistic.
- An alarm changes exposure from the next observation.
- Detector state resets at each evaluation period.
- The controller permits one persistent reduction
  per evaluation period.

## Backtesting

- Raw asset returns are multiplied by exposure once.
- Transaction costs are charged on exposure changes.
- Validation inherits the final training position.
- Training begins with an assumed flat position.
- Final test initialization must use the preceding
  validation position.

## Primary Evaluation Metrics

- Total compounded return
- Maximum drawdown
- Annualized Sharpe ratio
- Total turnover
- Total transaction costs
- Alarm frequency

## Findings Before Final Testing

Experiment 29 produced no detector alarms across
eight asset-period evaluations.

The controlled strategy therefore matched its
uncontrolled baseline.

These results do not establish an economic benefit
from the calibrated risk controller.

## Limitations

- Simplified transaction-cost model
- No nonlinear market impact
- No portfolio-weight drift
- No financing or borrowing costs
- Limited asset diversification
- Detector reset at period boundaries
- No confirmed historical deterioration labels
- Detector parameters calibrated using SPY
- Cross-asset evaluations are not fully independent

## Final Evaluation Rules

1. Do not modify frozen parameters using test results.
2. Do not optimize the strategy on the test period.
3. Verify all accounting and timing assumptions.
4. Run the final evaluation only after code review.
5. Report positive, negative, and null results.
6. Record dataset provenance and software versions.
7. Treat the final test as a one-time evaluation.
