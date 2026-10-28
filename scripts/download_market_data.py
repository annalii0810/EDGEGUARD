
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
import yfinance as yf


TICKERS = ["SPY", "QQQ", "IWM", "TLT"]
START_DATE = "2010-01-01"
END_DATE = "2026-10-28"

RAW_DIR = Path("data/raw")


def download_market_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE,
        "price_adjustment": "auto_adjust=True",
        "tickers": {},
    }

    for ticker in TICKERS:
        print(f"\nDownloading {ticker}...")

        try:
            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                actions=False,
                threads=False,
            )

            if data.empty:
                raise ValueError("No data returned.")

            # Handle yfinance MultiIndex columns.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if "Close" not in data.columns:
                raise ValueError("Missing Close column.")

            prices = data[["Close"]].copy()
            prices.index.name = "Date"
            prices = prices.reset_index()

            prices = prices[["Date", "Close"]]

            if prices["Close"].isna().any():
                raise ValueError("Missing adjusted prices.")

            if (prices["Close"] <= 0).any():
                raise ValueError("Invalid adjusted prices.")

            output = RAW_DIR / f"{ticker}.csv"
            prices.to_csv(output, index=False)

            manifest["tickers"][ticker] = {
                "file": str(output),
                "rows": len(prices),
                "first_date": str(prices["Date"].iloc[0]),
                "last_date": str(prices["Date"].iloc[-1]),
                "status": "success",
            }

            print(f"Saved {len(prices)} rows to {output}")

        except Exception as error:
            manifest["tickers"][ticker] = {
                "status": "failed",
                "error": str(error),
            }
            print(f"Failed: {error}")

    manifest_path = RAW_DIR / "download_manifest.json"

    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)

    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    download_market_data()
