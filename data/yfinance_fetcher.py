"""
YFinance Data Fetcher for MNQ Trading System

Fetches live and historical data for MNQ and correlation ring assets.

Note: yfinance limits 15m data to 60 days. For longer history, we use 1h data.
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CORRELATION_ASSETS = {
    "MNQ": "MNQ=F",  # Primary (Micro NQ)
    "ES": "ES=F",  # S&P 500
    "NQ": "NQ=F",  # Nasdaq 100
    "RTY": "RTY=F",  # Russell 2000
    "CL": "CL=F",  # Crude Oil
    "GC": "GC=F",  # Gold
    "ZB": "ZB=F",  # Bonds
    "6E": "6E=F",  # Euro
    "FDAX": "FDAX=F",  # DAX (Eurex)
}


def fetch_data(
    symbol: str,
    period: str = "60d",
    interval: str = "15m",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance.

    Args:
        symbol: yfinance ticker symbol (e.g., "MNQ=F")
        period: Data period (e.g., "60d", "8mo", "1y")
        interval: Bar interval (e.g., "15m", "1h", "1d")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV columns
    """
    try:
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            logger.warning(f"No data received for {symbol}")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        if "Volume" in data.columns and data["Volume"].isna().all():
            data = data.drop("Volume", axis=1)

        logger.info(f"Fetched {len(data)} bars for {symbol}")
        return data

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_mnq_primary(period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """Fetch MNQ data as primary dataset"""
    return fetch_data("MNQ=F", period=period, interval=interval)


def fetch_correlation_assets(
    period: str = "60d",
    interval: str = "15m",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch all correlation ring assets.

    Returns:
        Dict mapping symbol to DataFrame
    """
    data = {}
    for symbol, ticker in CORRELATION_ASSETS.items():
        if symbol == "MNQ":
            continue
        df = fetch_data(ticker, period=period, interval=interval)
        if not df.empty:
            data[symbol] = df
    return data


def fetch_all_for_training(
    period: str = "8mo",
    interval: str = "1h",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch MNQ and all correlation assets for training.

    Uses 1h bars for long history (8mo) due to yfinance 60-day limit on 15m.

    Args:
        period: Data period (default 8mo)
        interval: Bar interval (default 1h for long history)

    Returns:
        Dict with 'mnq' and 'correlations' keys
    """
    logger.info(f"Fetching {period} of {interval} data from yfinance...")
    logger.info(
        "Note: Using 1h bars for historical data (yfinance 15m limit = 60 days)"
    )

    mnq_data = fetch_mnq_primary(period=period, interval=interval)
    corr_data = fetch_correlation_assets(period=period, interval=interval)

    if mnq_data.empty:
        logger.error("Failed to fetch MNQ data!")
        return {"mnq": pd.DataFrame(), "correlations": {}}

    return {
        "mnq": mnq_data,
        "correlations": corr_data,
    }


def save_training_data(
    data: Dict[str, pd.DataFrame],
    base_path: str = "data",
) -> None:
    """
    Save fetched data to CSV files for training.

    Args:
        data: Dict from fetch_all_for_training
        base_path: Base directory for saving
    """
    mnq_path = f"{base_path}/mnq_yfinance.csv"
    data["mnq"].to_csv(mnq_path)
    logger.info(f"Saved MNQ data to {mnq_path}")

    for symbol, df in data["correlations"].items():
        corr_path = f"{base_path}/{symbol.lower()}_yfinance.csv"
        df.to_csv(corr_path)
        logger.info(f"Saved {symbol} data to {corr_path}")


if __name__ == "__main__":
    print("Fetching MNQ and correlation data from yfinance...")
    data = fetch_all_for_training(period="8mo", interval="1h")

    if data["mnq"].empty:
        print("ERROR: Failed to fetch MNQ data")
    else:
        print(f"\nMNQ: {len(data['mnq'])} bars")
        print(f"Date range: {data['mnq'].index[0]} to {data['mnq'].index[-1]}")

        print(f"\nCorrelation assets:")
        for symbol, df in data["correlations"].items():
            print(f"  {symbol}: {len(df)} bars")

        save_training_data(data)
        print("\nData saved to data/ directory")
