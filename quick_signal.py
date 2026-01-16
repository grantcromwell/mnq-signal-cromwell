#!/usr/bin/env python3
"""
Quick Discount Signal Script

Usage: python quick_signal.py

Outputs:
- Binary YES/NO signal
- Confidence level
- Price vs discount level
- Weekly Fibonacci levels
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf


def calculate_weekly_fib_info(close: pd.Series) -> dict:
    """
    Calculate weekly Fibonacci levels for discount detection.

    Returns dict with:
    - weekly_high: Highest price of current week
    - weekly_low: Lowest price of current week
    - fib_05: 0.5 Fib level (discount boundary)
    - fib_discount_05: 0.5 Fib of discount range (midpoint between fib_05 and weekly_low)
    """
    if close.empty:
        raise ValueError("Data is empty")

    dates = (
        close.index.to_pydatetime()
        if hasattr(close.index, "to_pydatetime")
        else close.index
    )

    current_week_start = None
    current_week_high = -np.inf
    current_week_low = np.inf

    for date, price in zip(dates, close.values):
        if current_week_start is None:
            current_week_start = date
            current_week_high = price
            current_week_low = price
        else:
            week_changed = (
                date.year != current_week_start.year
                or date.isocalendar()[1] != current_week_start.isocalendar()[1]
            )

            if week_changed:
                current_week_start = date
                current_week_high = price
                current_week_low = price
            else:
                current_week_high = max(current_week_high, price)
                current_week_low = min(current_week_low, price)

    weekly_range = current_week_high - current_week_low

    fib_05 = current_week_high - weekly_range * 0.5
    fib_discount_05 = current_week_low + weekly_range * 0.25

    return {
        "weekly_high": current_week_high,
        "weekly_low": current_week_low,
        "weekly_range": weekly_range,
        "fib_05": fib_05,
        "fib_discount_05": fib_discount_05,
    }


def get_quick_signal():
    """Get quick binary signal with confidence using live data."""

    print("Fetching live MNQ data...")
    data = yf.download("MNQ=F", period="5d", interval="15m", progress=False)

    if data.empty:
        print("ERROR: No data received from yfinance")
        return None

    close = data["Close"].squeeze()
    current_price = float(close.iloc[-1])

    fib_info = calculate_weekly_fib_info(close)

    fib_level = fib_info["fib_05"]
    in_discount = current_price < fib_level
    price_diff_pct = (fib_level - current_price) / fib_level * 100

    confidence = 0.80 if in_discount else 0.20
    signal = "YES" if in_discount else "NO"

    print("\n" + "=" * 55)
    print("MNQ DISCOUNT SIGNAL (LIVE)")
    print("=" * 55)
    print(f"Current Price:      {current_price:,.2f}")
    print("-" * 55)
    print(f"Weekly High:        {fib_info['weekly_high']:,.2f}")
    print(f"Weekly Low:         {fib_info['weekly_low']:,.2f}")
    print("-" * 55)
    print(f"Discount (0.5 Fib): {fib_level:,.2f}  <-- BUY below this")
    print(f"Discount Midpoint:  {fib_info['fib_discount_05']:,.2f}")
    print("-" * 55)
    print(f"Distance to Disc:   {price_diff_pct:+.2f}%")
    print(f"In Discount Zone:   {'YES' if in_discount else 'NO'}")
    print("=" * 55)
    print(f"  >>> SIGNAL: {signal}")
    print(f"  >>> CONFIDENCE: {confidence:.0%}")
    print("=" * 55)

    return {
        "signal": signal,
        "confidence": confidence,
        "price": current_price,
        "weekly_high": fib_info["weekly_high"],
        "weekly_low": fib_info["weekly_low"],
        "fib_05": fib_info["fib_05"],
        "fib_discount_05": fib_info["fib_discount_05"],
        "in_discount": in_discount,
        "distance_pct": price_diff_pct,
    }


if __name__ == "__main__":
    get_quick_signal()
