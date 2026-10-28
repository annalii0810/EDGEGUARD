# EDGEGUARD — Reproducibility Record

## Dataset Provenance

- Source: Yahoo Finance via yfinance
- Downloaded UTC: 2026-10-28T21:01:26.766580+00:00
- Start date: 2010-01-01
- End date (exclusive): 2026-10-28
- Price adjustment: auto_adjust=True

## Dataset SHA-256 Fingerprints

| Asset | Rows | SHA-256 |
|---|---:|---|
| SPY | 4230 | `17ecf817c040d2d1fc68bd5716bb55a185862f474b1be56806de6a4f94beb95b` |
| QQQ | 4230 | `b410637c245796f9c4f4920f82b78510022213435b88ff588f09599cf858883f` |
| IWM | 4230 | `b4cca6c30eedbd398d37fdb700bac8818ca05bdeb4900f089c9dc4e381be612c` |
| TLT | 4230 | `c92da42fa9305263584693ca89a40e6773608d3815a15f935b8ab34d3aa157a2` |

## Software Versions

- Python: 3.14.7
- numpy: 2.5.4
- pandas: 3.1.0
- yfinance: 1.7.0
- pytest: 9.1.1

## Evaluation Policy

- Training: 2010–2018
- Validation: 2019–2022
- Reserved test: 2023–2026
- Detector parameters frozen before final evaluation.
- Final test returns not used for parameter selection.

## Reproduction Notes

- Raw CSV datasets are excluded from Git.
- Download script: scripts/download_market_data.py
- Historical data providers may revise adjusted prices.
- Exact dataset fingerprints are recorded above.
