"""
Discount Labeler for MNQ Trading System

Labels each 15-min bar based on whether it represents a successful
discount opportunity (price below weekly 0.5 Fib that reaches 3:1 target).

Label Schema:
- 1: SUCCESS - Price in discount zone and reaches 3:1 target before stop
- 0: FAILURE - Price in discount zone but fails to reach target
- -1: NO_TRADE - Price in premium zone (above 0.5 Fib)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeLabel:
    """Trade outcome label"""

    bar_index: int
    timestamp: pd.Timestamp
    label: int  # 1=success, 0=failure, -1=no_trade
    entry_price: float
    stop_loss: float
    target_3r: float
    outcome: str  # "WIN", "LOSS", "FLAT", "NO_TRADE"
    bars_held: int
    pnl: float
    rr_achieved: float
    discount_level: float
    reason: str


@dataclass
class LabelingConfig:
    """Configuration for labeling strategy"""

    fib_level: float = 0.5
    atr_period: int = 14
    stop_atr_multiple: float = 0.5
    reward_risk_ratio: float = 3.0
    max_bars_hold: int = 10
    min_consecutive_discount_bars: int = 3
    confidence_threshold: float = 0.80


class DiscountLabeler:
    """
    Labels MNQ bars based on discount detection success.
    """

    def __init__(self, config: Optional[LabelingConfig] = None):
        self.config = config or LabelingConfig()

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR for stop loss determination"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.config.atr_period, min_periods=1).mean()

        return atr

    def calculate_weekly_fib(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weekly 0.5 Fibonacci levels"""
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "Unnamed: 0" in df.columns:
                df["time"] = pd.to_datetime(df["Unnamed: 0"])
                df = df.drop("Unnamed: 0", axis=1)
                df.set_index("time", inplace=True)
            else:
                logger.warning("No datetime column found, using index as datetime")
                df.index = pd.to_datetime(df.index)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        weekly = df.resample("W-SUN").agg(
            {
                "high": "max",
                "low": "min",
            }
        )

        weekly["weekly_range"] = weekly["high"] - weekly["low"]
        weekly["fib_05"] = weekly["low"] + (
            weekly["weekly_range"] * self.config.fib_level
        )

        fib_series = pd.Series(index=df.index, dtype=float)

        current_fib = None
        current_week_end = None

        for idx in df.index:
            week_end = idx + pd.Timedelta(days=(6 - idx.weekday()))

            if week_end != current_week_end:
                if week_end in weekly.index:
                    current_fib = (
                        weekly.loc[week_end, "fib_05"]
                        if not pd.isna(weekly.loc[week_end, "fib_05"])
                        else current_fib
                    )
                    current_week_end = week_end

            if current_fib is not None:
                fib_series.loc[idx] = current_fib

        return fib_series

    def is_in_discount_zone(self, price: float, fib_level: float) -> bool:
        """Check if price is below 0.5 Fib (discount zone)"""
        return price < fib_level

    def label_bar(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        fib_series: pd.Series,
        atr_series: pd.Series,
    ) -> TradeLabel:
        """
        Label a single bar's discount opportunity.

        Args:
            df: OHLCV DataFrame
            bar_idx: Index of bar to label
            fib_series: Weekly 0.5 Fib levels
            atr_series: ATR values

        Returns:
            TradeLabel with outcome
        """
        current_bar = df.iloc[bar_idx]
        current_price = current_bar["close"]
        current_time = df.index[bar_idx]

        fib_level = fib_series.iloc[bar_idx] if bar_idx < len(fib_series) else None

        if fib_level is None or pd.isna(fib_level):
            return TradeLabel(
                bar_index=bar_idx,
                timestamp=current_time,
                label=-1,
                entry_price=current_price,
                stop_loss=0,
                target_3r=0,
                outcome="NO_TRADE",
                bars_held=0,
                pnl=0,
                rr_achieved=0,
                discount_level=0,
                reason="No Fib level available",
            )

        in_discount = self.is_in_discount_zone(current_price, fib_level)

        if not in_discount:
            return TradeLabel(
                bar_index=bar_idx,
                timestamp=current_time,
                label=-1,
                entry_price=current_price,
                stop_loss=0,
                target_3r=0,
                outcome="NO_TRADE",
                bars_held=0,
                pnl=0,
                rr_achieved=0,
                discount_level=fib_level,
                reason="Premium zone (price >= 0.5 Fib)",
            )

        atr = atr_series.iloc[bar_idx] if bar_idx < len(atr_series) else 61.4
        if pd.isna(atr):
            atr = 61.4

        stop_distance = atr * self.config.stop_atr_multiple
        entry_price = current_price
        stop_loss = entry_price - stop_distance
        target_3r = entry_price + (stop_distance * self.config.reward_risk_ratio)

        outcome = "FLAT"
        label = 0
        bars_held = 0
        pnl = 0
        rr_achieved = 0

        for future_idx in range(
            bar_idx + 1, min(bar_idx + self.config.max_bars_hold + 1, len(df))
        ):
            future_bar = df.iloc[future_idx]
            future_high = future_bar["high"]
            future_low = future_bar["low"]

            bars_held = future_idx - bar_idx

            if future_high >= target_3r:
                outcome = "WIN"
                label = 1
                pnl = (target_3r - entry_price) * 0.5
                rr_achieved = self.config.reward_risk_ratio
                break

            if future_low <= stop_loss:
                outcome = "LOSS"
                label = 0
                pnl = -(entry_price - stop_loss) * 0.5
                rr_achieved = 0
                break

        reason_map = {
            "WIN": f"Reached 3:1 target at {target_3r:.0f} in {bars_held} bars",
            "LOSS": f"Hit stop at {stop_loss:.0f} in {bars_held} bars",
            "FLAT": f"No outcome in {self.config.max_bars_hold} bars",
        }

        return TradeLabel(
            bar_index=bar_idx,
            timestamp=current_time,
            label=label,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_3r=target_3r,
            outcome=outcome,
            bars_held=bars_held,
            pnl=pnl,
            rr_achieved=rr_achieved,
            discount_level=fib_level,
            reason=reason_map.get(outcome, "Unknown"),
        )

    def label_all_bars(self, df: pd.DataFrame) -> List[TradeLabel]:
        """
        Label all bars in the dataset.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of TradeLabel objects
        """
        logger.info(f"Labeling {len(df)} bars...")

        atr_series = self.calculate_atr(df)
        fib_series = self.calculate_weekly_fib(df)

        labels = []
        for bar_idx in range(len(df)):
            label = self.label_bar(df, bar_idx, fib_series, atr_series)
            labels.append(label)

            if (bar_idx + 1) % 1000 == 0:
                logger.info(f"Progress: {bar_idx + 1}/{len(df)} bars labeled")

        return labels

    def create_labeled_dataset(
        self,
        df: pd.DataFrame,
        include_features: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create a labeled dataset for training.

        Args:
            df: OHLCV DataFrame
            include_features: Include engineered features

        Returns:
            Tuple of (feature DataFrame, labels array)
        """
        labels = self.label_all_bars(df)

        label_df = pd.DataFrame(
            [
                {
                    "bar_index": l.bar_index,
                    "timestamp": l.timestamp,
                    "label": l.label,
                    "entry_price": l.entry_price,
                    "stop_loss": l.stop_loss,
                    "target_3r": l.target_3r,
                    "outcome": l.outcome,
                    "bars_held": l.bars_held,
                    "pnl": l.pnl,
                    "rr_achieved": l.rr_achieved,
                    "discount_level": l.discount_level,
                    "reason": l.reason,
                }
                for l in labels
            ]
        )

        if not include_features:
            return label_df, label_df["label"].values

        atr_series = self.calculate_atr(df)
        fib_series = self.calculate_weekly_fib(df)

        feature_df = label_df.copy()

        feature_df["price"] = df["close"].values
        feature_df["fib_distance_pts"] = fib_series.values - df["close"].values
        feature_df["fib_distance_pct"] = (
            feature_df["fib_distance_pts"] / fib_series.values
        )
        feature_df["price_vs_fib"] = df["close"].values / fib_series.values
        feature_df["is_discount"] = (df["close"].values < fib_series.values).astype(int)

        feature_df["momentum_3d"] = (
            df["close"].pct_change(12).values
        )  # 12 x 15min = 3 hours
        feature_df["momentum_1d"] = (
            df["close"].pct_change(96).values
        )  # 96 x 15min = 24 hours

        rsi = self._calculate_rsi(df["close"], 14)
        feature_df["rsi_14"] = rsi.values

        vol_ma = df["volume"].rolling(20).mean()
        feature_df["volume_ratio"] = df["volume"].values / vol_ma.values

        feature_df["atr"] = atr_series.values
        feature_df["volatility"] = df["close"].rolling(20).std().values

        hours = (
            df.index.hour
            if isinstance(df.index, pd.DatetimeIndex)
            else 12  # Default to noon for non-datetime index
        )
        feature_df["hour_of_day"] = hours
        feature_df["day_of_week"] = (
            df.index.dayofweek
            if isinstance(df.index, pd.DatetimeIndex)
            else 0  # Default to Monday for non-datetime index
        )

        feature_df["distance_to_weekly_high"] = (
            df["high"].rolling(96).max() - df["close"]
        ).values
        feature_df["distance_to_weekly_low"] = (
            df["close"] - df["low"].rolling(96).min()
        ).values

        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        feature_df = feature_df.fillna(0)

        feature_names = [
            "fib_distance_pts",
            "fib_distance_pct",
            "price_vs_fib",
            "is_discount",
            "momentum_3d",
            "momentum_1d",
            "rsi_14",
            "volume_ratio",
            "atr",
            "volatility",
            "hour_of_day",
            "day_of_week",
            "distance_to_weekly_high",
            "distance_to_weekly_low",
        ]

        X = feature_df[feature_names].values
        y = label_df["label"].values

        return feature_df, X, y

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def get_label_summary(self, labels: List[TradeLabel]) -> Dict[str, Any]:
        """
        Get summary statistics for labels.

        Args:
            labels: List of TradeLabel objects

        Returns:
            Dictionary with summary statistics
        """
        df = pd.DataFrame(
            [
                {
                    "label": l.label,
                    "outcome": l.outcome,
                    "pnl": l.pnl,
                    "rr_achieved": l.rr_achieved,
                    "bars_held": l.bars_held,
                }
                for l in labels
            ]
        )

        total_trades = len(df[df["label"] != -1])
        successes = len(df[df["label"] == 1])
        failures = len(df[df["label"] == 0])
        no_trades = len(df[df["label"] == -1])

        return {
            "total_bars": len(labels),
            "total_trades": total_trades,
            "successes": successes,
            "failures": failures,
            "no_trades": no_trades,
            "win_rate": successes / total_trades if total_trades > 0 else 0,
            "avg_pnl": df[df["label"] != -1]["pnl"].mean(),
            "total_pnl": df[df["label"] != -1]["pnl"].sum(),
            "avg_bars_held": df[df["label"] != -1]["bars_held"].mean(),
        }


if __name__ == "__main__":
    np.random.seed(42)
    n = 5000
    base_price = 17500

    dates = pd.date_range("2025-03-01", periods=n, freq="15min")
    closes = pd.Series(
        base_price * (1 + np.cumsum(np.random.randn(n) * 0.0003)), index=dates
    )
    highs = closes + np.random.rand(n) * 15
    lows = closes - np.random.rand(n) * 15
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

    labeler = DiscountLabeler()
    labels = labeler.label_all_bars(df)

    summary = labeler.get_label_summary(labels)
    print("=== Label Summary ===")
    print(f"Total bars: {summary['total_bars']}")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Successes: {summary['successes']}")
    print(f"Failures: {summary['failures']}")
    print(f"No trades: {summary['no_trades']}")
    print(f"Win rate: {summary['win_rate']:.1%}")
    print(f"Avg PnL: ${summary['avg_pnl']:.2f}")
    print(f"Total PnL: ${summary['total_pnl']:.2f}")

    feature_df, X, y = labeler.create_labeled_dataset(df)
    print(f"\n=== Dataset ===")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y + 1)}")
