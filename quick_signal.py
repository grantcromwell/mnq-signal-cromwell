#!/usr/bin/env python3
"""
Quick Discount Signal Script

Usage: python quick_signal.py

Outputs:
- Binary YES/NO signal
- Confidence level
- Price vs discount level
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf


def calculate_weekly_fib(close: pd.Series) -> float:
    """Calculate weekly 0.5 Fibonacci level for current week."""
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

    fib_level = current_week_high - (current_week_high - current_week_low) * 0.5
    return fib_level


def get_quick_signal():
    """Get quick binary signal with confidence using live data."""

    print("Fetching live MNQ data...")
    data = yf.download("MNQ=F", period="5d", interval="15m", progress=False)

    if data.empty:
        print("ERROR: No data received from yfinance")
        return None

    close = data["Close"].squeeze()
    current_price = float(close.iloc[-1])

    fib_level = calculate_weekly_fib(close)

    in_discount = current_price < fib_level
    price_diff_pct = (fib_level - current_price) / fib_level * 100

    confidence = 0.80 if in_discount else 0.20
    signal = "YES" if in_discount else "NO"

    print("\n" + "=" * 50)
    print("MNQ DISCOUNT SIGNAL (LIVE)")
    print("=" * 50)
    print(f"Current Price:    {current_price:,.2f}")
    print(f"Weekly 0.5 Fib:   {fib_level:,.2f}")
    print(f"Distance:         {price_diff_pct:+.2f}%")
    print(f"In Discount:      {'YES' if in_discount else 'NO'}")
    print("-" * 50)
    print(f"  >>> SIGNAL: {signal}")
    print(f"  >>> CONFIDENCE: {confidence:.0%}")
    print("=" * 50)

    return {
        "signal": signal,
        "confidence": confidence,
        "price": current_price,
        "fib_level": fib_level,
        "in_discount": in_discount,
        "distance_pct": price_diff_pct,
    }


if __name__ == "__main__":
    get_quick_signal()
