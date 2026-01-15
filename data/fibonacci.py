"""
Weekly Fibonacci Level Calculator

Calculates the 0.5 Fibonacci retracement level for each week
to identify discount (below 0.5 Fib) vs premium (above 0.5 Fib) zones.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeeklyFibLevel:
    """Weekly Fibonacci level data"""

    week_start: pd.Timestamp
    week_end: pd.Timestamp
    weekly_high: float
    weekly_low: float
    fib_05_level: float
    fib_0382_level: float
    fib_0618_level: float
    is_discount: bool  # True if current price < 0.5 Fib


@dataclass
class DiscountSignal:
    """Discount zone signal for a bar"""

    timestamp: pd.Timestamp
    is_discount: bool
    fib_05_level: float
    price: float
    distance_to_fib: float
    distance_pct: float
    confidence: float


class FibonacciCalculator:
    """
    Calculates weekly Fibonacci retracement levels and
    identifies discount/premium zones.
    """

    def __init__(self, fib_level: float = 0.5):
        """
        Args:
            fib_level: Fibonacci level to use (0.5 = midpoint)
        """
        self.fib_level = fib_level

    def calculate_weekly_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weekly OHLC and Fibonacci levels.

        Args:
            df: DataFrame with 'high', 'low', 'close', 'open' columns

        Returns:
            DataFrame with weekly levels and Fib calculations
        """
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

        weekly = df.resample("W-SUN").agg(
            {
                "high": "max",
                "low": "min",
                "close": "last",
                "open": "first",
                "volume": "sum",
            }
        )

        weekly["weekly_range"] = weekly["high"] - weekly["low"]
        weekly[f"fib_{self.fib_level}"] = weekly["low"] + (
            weekly["weekly_range"] * self.fib_level
        )
        weekly["fib_0382"] = weekly["low"] + (weekly["weekly_range"] * 0.382)
        weekly["fib_0618"] = weekly["low"] + (weekly["weekly_range"] * 0.618)

        weekly["week_start"] = weekly.index
        weekly["week_end"] = weekly.index + pd.Timedelta(days=6, hours=23, minutes=59)

        weekly = weekly.dropna()

        return weekly

    def get_current_fib_level(
        self, df: pd.DataFrame, current_time: pd.Timestamp
    ) -> Optional[float]:
        """
        weekly        Get the current 0.5 Fib level for a given time.

               Args:
                   df: DataFrame with weekly levels
                   current_time: Current timestamp

               Returns:
                   0.5 Fib level or None if not found
        """
        weekly = self.calculate_weekly_levels(df)

        for idx, row in weekly.iterrows():
            if idx <= current_time:
                return row[f"fib_{self.fib_level}"]

        return None

    def is_in_discount_zone(
        self,
        price: float,
        fib_05_level: float,
    ) -> bool:
        """
        Check if price is in discount zone (below 0.5 Fib).

        Args:
            price: Current price
            fib_05_level: Weekly 0.5 Fib level

        Returns:
            True if price < 0.5 Fib (discount zone)
        """
        return price < fib_05_level

    def calculate_distance_to_fib(
        self,
        price: float,
        fib_05_level: float,
        atr: float = None,
    ) -> Tuple[float, float]:
        """
        Calculate distance from price to 0.5 Fib level.

        Args:
            price: Current price
            fib_05_level: Weekly 0.5 Fib level
            atr: ATR for normalization (optional)

        Returns:
            Tuple of (distance in points, distance as % of Fib level)
        """
        distance_pts = fib_05_level - price
        distance_pct = distance_pts / fib_05_level if fib_05_level > 0 else 0

        return distance_pts, distance_pct

    def get_discount_features(
        self,
        df: pd.DataFrame,
        current_idx: int,
        weekly_levels: pd.DataFrame = None,
    ) -> Dict[str, float]:
        """
        Calculate all features related to discount detection.

        Args:
            df: 15-min bar data
            current_idx: Current bar index
            weekly_levels: Pre-calculated weekly levels

        Returns:
            Dictionary of discount features
        """
        if current_idx < 0 or current_idx >= len(df):
            return {}

        current_price = df["close"].iloc[current_idx]
        current_time = df.index[current_idx]

        if weekly_levels is None:
            weekly_levels = self.calculate_weekly_levels(df)

        fib_05 = self.get_current_fib_level(df, current_time)

        if fib_05 is None:
            return {}

        is_discount = self.is_in_discount_zone(current_price, fib_05)
        distance_pts, distance_pct = self.calculate_distance_to_fib(
            current_price, fib_05
        )

        recent_high = df["high"].iloc[max(0, current_idx - 3) : current_idx + 1].max()
        recent_low = df["low"].iloc[max(0, current_idx - 3) : current_idx + 1].min()

        features = {
            "fib_05_level": fib_05,
            "fib_distance_pts": distance_pts,
            "fib_distance_pct": distance_pct,
            "is_discount": float(is_discount),
            "distance_to_weekly_high": recent_high - current_price,
            "distance_to_weekly_low": current_price - recent_low,
            "price_vs_fib": current_price / fib_05 if fib_05 > 0 else 1.0,
        }

        return features

    def identify_discount_zones(
        self,
        df: pd.DataFrame,
        min_bars: int = 5,
    ) -> List[DiscountSignal]:
        """
        Identify all discount zone signals in the data.

        Args:
            df: 15-min bar data
            min_bars: Minimum consecutive bars in discount to signal

        Returns:
            List of DiscountSignal objects
        """
        weekly_levels = self.calculate_weekly_levels(df)
        signals = []

        current_fib = None
        consecutive_discount = 0

        for idx, row in df.iterrows():
            price = row["close"]
            current_time = idx

            fib_05 = self.get_current_fib_level(df, current_time)

            if fib_05 is None:
                continue

            is_discount = self.is_in_discount_zone(price, fib_05)
            distance_pts, distance_pct = self.calculate_distance_to_fib(price, fib_05)

            if is_discount:
                consecutive_discount += 1
            else:
                consecutive_discount = 0

            if fib_05 != current_fib:
                current_fib = fib_05

            if is_discount and consecutive_discount >= min_bars:
                confidence = min(
                    0.95, 0.60 + (distance_pct * 10) + (consecutive_discount * 0.02)
                )
            else:
                confidence = 0.0

            signals.append(
                DiscountSignal(
                    timestamp=current_time,
                    is_discount=is_discount,
                    fib_05_level=fib_05,
                    price=price,
                    distance_to_fib=distance_pts,
                    distance_pct=distance_pct,
                    confidence=confidence,
                )
            )

        return signals


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        df: DataFrame with 'high', 'low', 'close'
        period: ATR period

    Returns:
        Series with ATR values
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    return atr


def calculate_volume_profile(
    df: pd.DataFrame,
    price_bins: int = 50,
) -> Dict[str, float]:
    """
    Calculate volume profile features for additional context.

    Args:
        df: OHLCV DataFrame
        price_bins: Number of price levels

    Returns:
        Dictionary with volume profile features
    """
    high_max = df["high"].max()
    low_min = df["low"].min()
    price_range = high_max - low_min
    bin_size = price_range / price_bins

    price_centers = np.linspace(
        low_min + bin_size / 2,
        high_max - bin_size / 2,
        price_bins,
    )

    volume_at_price = np.zeros(price_bins)

    for i in range(len(df)):
        high_idx = min(int((df["high"].iloc[i] - low_min) / bin_size), price_bins - 1)
        low_idx = min(int((df["low"].iloc[i] - low_min) / bin_size), price_bins - 1)
        vol = df["volume"].iloc[i] if "volume" in df.columns else 1

        avg_vol = vol / max(high_idx - low_idx + 1, 1)
        volume_at_price[low_idx : high_idx + 1] += avg_vol

    poc_idx = np.argmax(volume_at_price)
    poc = price_centers[poc_idx]

    cumsum = np.cumsum(volume_at_price)
    val_area_low_idx = np.searchsorted(cumsum, (1 - 0.70) / 2)
    val_area_high_idx = np.searchsorted(cumsum, (1 + 0.70) / 2)

    value_area_high = price_centers[min(val_area_high_idx, len(price_centers) - 1)]
    value_area_low = price_centers[max(val_area_low_idx, 0)]

    vol_above_poc = volume_at_price[price_centers > poc].sum()
    vol_below_poc = volume_at_price[price_centers < poc].sum()

    return {
        "poc": poc,
        "value_area_high": value_area_high,
        "value_area_low": value_area_low,
        "volume_asymmetry": vol_above_poc / max(vol_below_poc, 0.001),
    }


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    base_price = 17500

    dates = pd.date_range("2025-03-01", periods=n, freq="15min")
    closes = pd.Series(
        base_price * (1 + np.cumsum(np.random.randn(n) * 0.0005)), index=dates
    )
    highs = closes + np.random.rand(n) * 20
    lows = closes - np.random.rand(n) * 20
    volumes = pd.Series(np.random.randint(10000, 50000, n), index=dates)

    df = pd.DataFrame(
        {
            "open": closes.shift(1).fillna(base_price),
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    fib_calc = FibonacciCalculator(fib_level=0.5)
    weekly = fib_calc.calculate_weekly_levels(df)

    print("=== Weekly Fibonacci Levels ===")
    print(weekly[["weekly_high", "weekly_low", "fib_0.5"]].tail(10))

    print("\n=== Discount Zone Detection ===")
    signals = fib_calc.identify_discount_zones(df)
    discount_signals = [s for s in signals if s.is_discount]
    print(f"Total bars: {len(signals)}")
    print(f"Discount zone bars: {len(discount_signals)}")

    latest = signals[-1]
    print(f"\nLatest Bar:")
    print(f"  Time: {latest.timestamp}")
    print(f"  Price: {latest.price:.0f}")
    print(f"  0.5 Fib: {latest.fib_05_level:.0f}")
    print(f"  In Discount Zone: {latest.is_discount}")
    print(f"  Distance to Fib: {latest.distance_to_fib:.0f} pts")
