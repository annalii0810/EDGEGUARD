# EDGEGUARD: Evaluating Adaptive Change-Point Detection for Quantitative Trading Risk Management

**Research type:** Independent undergraduate quantitative research  
**Status:** Completed historical evaluation and exploratory synthetic experiments  
**Date:** October 2026

---

## Abstract

This study investigates whether adaptive change-point detection can identify deterioration in trading-strategy returns and improve risk-adjusted performance through dynamic exposure reduction.

EDGEGUARD combines an exposure-aware adaptive cumulative sum (CUSUM) detector with a risk controller that reduces trading exposure after detecting persistent negative deviations in strategy returns.

The research evaluates EDGEGUARD using historical exchange-traded fund data, controlled synthetic deterioration simulations, and comparisons against conventional statistical detectors.

In a frozen historical evaluation covering January 2023 through October 2026, EDGEGUARD generated no alarms across SPY, QQQ, IWM, and TLT. Consequently, the risk controller produced no measured improvement in returns, drawdowns, or Sharpe ratios.

Under controlled synthetic deterioration, EDGEGUARD improved mean total returns by 27.09 percentage points and reduced mean maximum drawdown by 24.09 percentage points relative to an unprotected strategy.

However, conventional standardized CUSUM slightly outperformed EDGEGUARD in direct synthetic economic comparisons.

The findings demonstrate that adaptive detection can support effective exposure reduction under specific simulated deterioration conditions, but do not establish superiority over conventional detectors or improved real-market trading performance.

## 1. Introduction

Quantitative trading strategies may experience deterioration when market conditions change, statistical relationships weaken, or previously profitable signals become ineffective.

Risk-management systems attempt to identify such changes and limit losses without unnecessarily reducing exposure during normal market conditions.

This creates a fundamental tradeoff between detection speed and false alarms.

An overly sensitive detector may reduce exposure unnecessarily, sacrificing profitable opportunities. A conservative detector may respond too slowly to prevent substantial losses.

EDGEGUARD investigates whether an adaptive, exposure-aware CUSUM detector can improve this tradeoff.

The research addresses three primary questions:

1. Can adaptive CUSUM detect persistent deterioration in trading-strategy returns?
2. Does reducing exposure following an alarm improve simulated economic performance?
3. Does adaptive volatility estimation provide advantages over conventional fixed-volatility detection?

## 2. Methodology

### 2.1 Historical market data

Historical adjusted closing prices were downloaded using Yahoo Finance through the `yfinance` Python package.

Four exchange-traded funds were examined:

- SPY — U.S. large-cap equities.
- QQQ — Nasdaq-100 equities.
- IWM — U.S. small-cap equities.
- TLT — Long-duration U.S. Treasury bonds.

The dataset contains daily observations beginning January 2010 and ending October 27, 2026.

Historical observations were divided chronologically into:

| Period | Dates | Purpose |
|---|---|---|
| Training | 2010–2018 | Reference estimation and development |
| Validation | 2019–2022 | Methodology evaluation |
| Final test | 2023–October