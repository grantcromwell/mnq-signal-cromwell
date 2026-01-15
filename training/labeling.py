"""
3:1 Risk/Reward Labeling Strategy

Defines how to label historical data for supervised training.
A trade is labeled as SUCCESS (1) if price reaches 3x the stop loss
distance before being stopped out.

Key concepts:
- Entry: Signal generation point
- Stop Loss: Predefined loss limit (e.g., 1 ATR below entry)
- Target 3R: Entry + 3 * (Entry - Stop Loss)
- Success: Price reaches Target 3R before Stop Loss
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeLabel:
    """Single trade label"""

    timestamp: datetime
    entry_price: float
    stop_loss: float
    target_3r: float
    success: bool
    max_r_reached: float
    bars_to_outcome: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_3r": self.target_3r,
            "success": self.success,
            "max_r_reached": self.max_r_reached,
            "bars_to_outcome": self.bars_to_outcome,
        }


@dataclass
class LabelingConfig:
    """Configuration for trade labeling"""

    risk_reward_ratio: float = 3.0
    atr_period: int = 14
    atr_multiplier_sl: float = 1.0
    max_bars_hold: int = 24
    session_start_hour: int = 9
    session_end_hour: int = 16


class ATRCalculator:
    """Average True Range calculator"""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ATR for DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            ATR Series
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=self.period).mean()

        return atr


class TradeSimulator:
    """
    Simulates trades and determines outcomes based on 3:1 R/R rule.
    """

    def __init__(self, config: Optional[LabelingConfig] = None):
        if config is None:
            config = LabelingConfig()
        self.config = config
        self.atr_calc = ATRCalculator(config.atr_period)

    def compute_stop_and_target(
        self, entry_price: float, atr: float, direction: str = "long"
    ) -> Tuple[float, float]:
        """
        Compute stop loss and 3R target.

        Args:
            entry_price: Trade entry price
            atr: Current ATR value
            direction: "long" or "short"

        Returns:
            Tuple of (stop_loss, target_3r)
        """
        sl_distance = atr * self.config.atr_multiplier_sl

        if direction == "long":
            stop_loss = entry_price - sl_distance
            target_3r = entry_price + (sl_distance * self.config.risk_reward_ratio)
        else:
            stop_loss = entry_price + sl_distance
            target_3r = entry_price - (sl_distance * self.config.risk_reward_ratio)

        return stop_loss, target_3r

    def simulate_long_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_loss: float,
        target_3r: float,
    ) -> TradeLabel:
        """
        Simulate a long trade outcome.

        Args:
            df: OHLCV DataFrame
            entry_idx: Index of entry bar
            entry_price: Entry price
            stop_loss: Stop loss level
            target_3r: 3R target level

        Returns:
            TradeLabel with outcome
        """
        future_bars = df.iloc[entry_idx + 1 :]

        if future_bars.empty:
            return TradeLabel(
                timestamp=df.index[entry_idx]
                if entry_idx < len(df)
                else datetime.now(),
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_3r=target_3r,
                success=False,
                max_r_reached=0.0,
                bars_to_outcome=-1,
            )

        entry_time = df.index[entry_idx]

        hit_stop = False
        hit_target = False
        max_r_reached = 0.0

        for i, (idx, row) in enumerate(future_bars.iterrows()):
            high = row["high"]
            low = row["low"]

            if low <= stop_loss:
                hit_stop = True
                bars_to_outcome = i + 1
                break

            if high >= target_3r:
                hit_target = True
                bars_to_outcome = i + 1
                break

            current_r = (
                (row["close"] - entry_price) / (entry_price - stop_loss)
                if entry_price > stop_loss
                else 0
            )
            max_r_reached = max(max_r_reached, current_r)

        else:
            bars_to_outcome = len(future_bars)
            hit_stop = False
            hit_target = False

        success = hit_target and not hit_stop

        return TradeLabel(
            timestamp=entry_time,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_3r=target_3r,
            success=success,
            max_r_reached=max_r_reached,
            bars_to_outcome=bars_to_outcome,
        )

    def simulate_short_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_loss: float,
        target_3r: float,
    ) -> TradeLabel:
        """Simulate a short trade outcome"""
        future_bars = df.iloc[entry_idx + 1 :]

        if future_bars.empty:
            return TradeLabel(
                timestamp=df.index[entry_idx]
                if entry_idx < len(df)
                else datetime.now(),
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_3r=target_3r,
                success=False,
                max_r_reached=0.0,
                bars_to_outcome=-1,
            )

        entry_time = df.index[entry_idx]

        hit_stop = False
        hit_target = False
        max_r_reached = 0.0

        for i, (idx, row) in enumerate(future_bars.iterrows()):
            high = row["high"]
            low = row["low"]

            if high >= stop_loss:
                hit_stop = True
                bars_to_outcome = i + 1
                break

            if low <= target_3r:
                hit_target = True
                bars_to_outcome = i + 1
                break

            current_r = (
                (entry_price - row["close"]) / (stop_loss - entry_price)
                if entry_price < stop_loss
                else 0
            )
            max_r_reached = max(max_r_reached, current_r)

        else:
            bars_to_outcome = len(future_bars)
            hit_stop = False
            hit_target = False

        success = hit_target and not hit_stop

        return TradeLabel(
            timestamp=entry_time,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_3r=target_3r,
            success=success,
            max_r_reached=max_r_reached,
            bars_to_outcome=bars_to_outcome,
        )


class TradeLabeler:
    """
    Main labeling class that processes price data and generates labels.
    """

    def __init__(self, config: Optional[LabelingConfig] = None):
        if config is None:
            config = LabelingConfig()
        self.config = config
        self.simulator = TradeSimulator(config)
        self.atr_calc = ATRCalculator(config.atr_period)

    def label_from_signals(
        self, df: pd.DataFrame, signal_indices: List[int], signal_directions: List[str]
    ) -> List[TradeLabel]:
        """
        Label trades from signal indices.

        Args:
            df: OHLCV DataFrame
            signal_indices: List of entry bar indices
            signal_directions: List of "long" or "short"

        Returns:
            List of TradeLabel objects
        """
        atr = self.atr_calc.compute(df)

        labels = []

        for idx, direction in zip(signal_indices, signal_directions):
            if idx >= len(df) - 1:
                continue

            entry_price = df["close"].iloc[idx]
            current_atr = atr.iloc[idx] if not atr.empty else (entry_price * 0.01)

            stop_loss, target_3r = self.simulator.compute_stop_and_target(
                entry_price, current_atr, direction
            )

            if direction == "long":
                label = self.simulator.simulate_long_trade(
                    df, idx, entry_price, stop_loss, target_3r
                )
            else:
                label = self.simulator.simulate_short_trade(
                    df, idx, entry_price, stop_loss, target_3r
                )

            labels.append(label)

        return labels

    def generate_all_labels(
        self, df: pd.DataFrame, direction: str = "long"
    ) -> List[TradeLabel]:
        """
        Generate labels for all potential entries.

        Labels bars where a long would have been successful.

        Args:
            df: OHLCV DataFrame
            direction: "long" or "short"

        Returns:
            List of TradeLabel for each bar
        """
        atr = self.atr_calc.compute(df)
        labels = []

        for idx in range(len(df) - 1):
            entry_price = df["close"].iloc[idx]
            current_atr = (
                atr.iloc[idx]
                if not atr.empty and not pd.isna(atr.iloc[idx])
                else (entry_price * 0.01)
            )

            stop_loss, target_3r = self.simulator.compute_stop_and_target(
                entry_price, current_atr, direction
            )

            if direction == "long":
                label = self.simulator.simulate_long_trade(
                    df, idx, entry_price, stop_loss, target_3r
                )
            else:
                label = self.simulator.simulate_short_trade(
                    df, idx, entry_price, stop_loss, target_3r
                )

            labels.append(label)

        return labels

    def get_binary_labels(
        self, df: pd.DataFrame, direction: str = "long"
    ) -> np.ndarray:
        """
        Get binary success/failure labels for all bars.

        Args:
            df: OHLCV DataFrame
            direction: "long" or "short"

        Returns:
            Binary numpy array (1 = success, 0 = failure)
        """
        labels = self.generate_all_labels(df, direction)
        return np.array([1 if l.success else 0 for l in labels])

    def get_label_statistics(self, labels: List[TradeLabel]) -> Dict[str, Any]:
        """
        Calculate statistics for a set of labels.

        Args:
            labels: List of TradeLabel objects

        Returns:
            Dict of statistics
        """
        if not labels:
            return {
                "total_trades": 0,
                "success_rate": 0.0,
                "avg_bars_to_outcome": 0.0,
                "avg_max_r": 0.0,
            }

        successes = [l for l in labels if l.success]
        failures = [l for l in labels if not l.success]

        stats = {
            "total_trades": len(labels),
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": len(successes) / len(labels) if labels else 0.0,
            "avg_bars_to_outcome": np.mean(
                [l.bars_to_outcome for l in labels if l.bars_to_outcome > 0]
            )
            if labels
            else 0.0,
            "avg_max_r": np.mean([l.max_r_reached for l in labels]) if labels else 0.0,
            "max_r_achieved": max([l.max_r_reached for l in labels]) if labels else 0.0,
        }

        return stats


def create_balanced_dataset(
    df: pd.DataFrame,
    labeler: TradeLabeler,
    ratio: float = 0.5,
    samples_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a balanced training dataset.

    Args:
        df: OHLCV DataFrame
        labeler: TradeLabeler instance
        ratio: Target ratio of positive samples
        samples_per_class: Maximum samples per class (optional)

    Returns:
        Dict with 'features' and 'labels'
    """
    labels = labeler.generate_all_labels(df, direction="long")
    binary_labels = np.array([1 if l.success else 0 for l in labels])

    success_indices = np.where(binary_labels == 1)[0]
    failure_indices = np.where(binary_labels == 0)[0]

    if samples_per_class is None:
        min_class_size = min(len(success_indices), len(failure_indices))
        n_samples = int(min_class_size * ratio) if ratio < 1.0 else min_class_size
    else:
        n_samples = min(samples_per_class, len(success_indices), len(failure_indices))

    selected_success = np.random.choice(success_indices, size=n_samples, replace=False)
    selected_failure = np.random.choice(failure_indices, size=n_samples, replace=False)

    selected_indices = np.concatenate([selected_success, selected_failure])
    np.random.shuffle(selected_indices)

    return {
        "indices": selected_indices,
        "labels": binary_labels[selected_indices],
        "trade_labels": [labels[i] for i in selected_indices],
    }


if __name__ == "__main__":
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=500, freq="15min")

    price = 17000.0
    prices = [price]
    for i in range(499):
        change = np.random.randn() * 20
        price += change
        prices.append(price)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p + abs(np.random.randn() * 10) for p in prices],
            "low": [p - abs(np.random.randn() * 10) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, 500),
        },
        index=dates,
    )

    labeler = TradeLabeler()
    labels = labeler.generate_all_labels(df, direction="long")

    stats = labeler.get_label_statistics(labels)
    print(f"Total trades: {stats['total_trades']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average bars to outcome: {stats['avg_bars_to_outcome']:.1f}")
    print(f"Average max R achieved: {stats['avg_max_r']:.2f}")
