# EDGEGUARD — Phase 5 Research Findings

## Research Objective

Investigate when adaptive change-point detection can provide economically meaningful protection against deterioration in trading-strategy returns.

## Historical Evidence

The frozen historical evaluation (2023–2026) generated no alarms across SPY, QQQ, IWM, and TLT. Consequently, EDGEGUARD did not improve measured returns, drawdowns, or Sharpe ratios.

These null findings are preserved without retrospective parameter changes.

## Synthetic Detection Results

Controlled deterioration simulations demonstrated high detection probability under severe, persistent negative mean shifts.

However, detection performance declined as volatility increased. At 2% simulated daily volatility, detection probability fell to 59.6%.

## Comparison Against Conventional Detectors

Independent paired simulations showed that EDGEGUARD did not consistently outperform rolling-mean detection or standardized CUSUM.

At a 50-observation detection deadline, EDGEGUARD achieved 39.0% detection, compared with 59.0% for Rolling Mean and 44.0% for Standard CUSUM.

At a 100-observation deadline, EDGEGUARD achieved 93.1%, compared with 88.2% and 94.2%, respectively.

These results are specific to the simulated deterioration scenario.

## Economic Risk-Control Results

Under severe synthetic deterioration, EDGEGUARD improved mean total return by 27.09 percentage points and reduced mean maximum drawdown by 24.09 percentage points relative to an unprotected strategy.

Under no deterioration, EDGEGUARD reduced mean total return by 0.25 percentage points.

A conventional CUSUM-based controller slightly outperformed EDGEGUARD under the tested deterioration scenario.

## Volatility Misspecification

Standard CUSUM's false-alarm probability varied substantially when its assumed volatility differed from the true simulation volatility.

At 50% of true volatility, the false-alarm rate reached 85.8%. At 150% and 200%, no false alarms were observed among 500 paths.

EDGEGUARD produced a 4.8% false-alarm rate under the same fixed data-generating volatility.

This demonstrates sensitivity to scale misspecification in the fixed-volatility baseline. It does not establish general superiority of EDGEGUARD.

## Limitations

- Synthetic returns were generated using simplified Gaussian distributions and persistent mean shifts.
- The historical evaluation showed no economic improvement.
- Baseline detectors sometimes received the true simulation volatility.
- Multiple exploratory experiments have been conducted; their results are not independent confirmatory evidence.
- Realistic volatility clustering, fat tails, regime changes, and parameter-estimation uncertainty require further investigation.
- Synthetic economic results do not establish live trading profitability.

## Provisional Conclusion

EDGEGUARD provides a reproducible framework for studying change-point detection and exposure-based risk control.

Its strongest demonstrated benefit is protection against severe, persistent deterioration in controlled simulations.

The current evidence does not establish superiority over conventional detectors or improved historical trading performance.

Further research should focus on realistic volatility dynamics, fair calibration, and the tradeoff between false alarms, detection delay, and economic risk reduction.