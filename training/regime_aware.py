"""
Regime-Aware Training Module for MNQ Trading Signal System

Implements regime detection and regime-aware training to improve
model performance across different market conditions.
"""

import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""

    BULL_STRONG = "bull_strong"
    BULL_WEAK = "bull_weak"
    BEAR_STRONG = "bear_strong"
    BEAR_WEAK = "bear_weak"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeData:
    """Container for regime-specific training data"""

    regime: MarketRegime
    features: np.ndarray
    labels: np.ndarray
    confidences: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimePerformance:
    """Performance metrics per regime"""

    regime: MarketRegime
    win_rate: float
    total_trades: int
    winning_trades: int
    avg_confidence: float
    avg_return: float
    profit_factor: float


class RegimeDetector:
    """
    Detects market regime based on price action and technical indicators.
    """

    def __init__(
        self,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.015,
        sideways_threshold: float = 0.01,
    ):
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.sideways_threshold = sideways_threshold

    def compute_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20,
    ) -> Dict[str, float]:
        """Compute regime detection features"""
        returns = close.pct_change()
        abs_returns = np.abs(returns)

        trend = (
            (close.iloc[-1] / close.iloc[-lookback]) - 1
            if len(close) >= lookback
            else 0
        )

        volatility = (
            returns.rolling(lookback).std().iloc[-1]
            if len(returns) >= lookback
            else 0.01
        )

        high_lookback = high.rolling(lookback).max()
        low_lookback = low.rolling(lookback).min()
        range_pct = (high_lookback - low_lookback) / close

        volume_ma = volume.rolling(lookback).mean()
        volume_ratio = (
            volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
        )

        direction = np.sign(trend)
        strength = abs(trend)

        return {
            "trend": trend,
            "volatility": volatility,
            "range_pct": range_pct.iloc[-1] if hasattr(range_pct, "iloc") else 0.05,
            "volume_ratio": volume_ratio,
            "direction": direction,
            "strength": strength,
        }

    def detect_regime(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20,
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Trading volume
            lookback: Number of bars to look back

        Returns:
            MarketRegime enum value
        """
        features = self.compute_features(high, low, close, volume, lookback)

        trend = features["trend"]
        volatility = features["volatility"]
        strength = features["strength"]

        if volatility > self.volatility_threshold * 1.5:
            return MarketRegime.VOLATILE

        if strength < self.sideways_threshold:
            return MarketRegime.SIDEWAYS

        if trend > self.trend_threshold:
            if trend > self.trend_threshold * 2:
                return MarketRegime.BULL_STRONG
            else:
                return MarketRegime.BULL_WEAK
        elif trend < -self.trend_threshold:
            if trend < -self.trend_threshold * 2:
                return MarketRegime.BEAR_STRONG
            else:
                return MarketRegime.BEAR_WEAK
        else:
            return MarketRegime.SIDEWAYS


class RegimeAwareSampler:
    """
    Sampling strategy that balances regime representation in training.
    """

    def __init__(
        self,
        target_distribution: Optional[Dict[MarketRegime, float]] = None,
        regime_detector: Optional[RegimeDetector] = None,
    ):
        self.regime_detector = regime_detector or RegimeDetector()
        self.target_distribution = target_distribution or {
            MarketRegime.BULL_STRONG: 0.15,
            MarketRegime.BULL_WEAK: 0.20,
            MarketRegime.SIDEWAYS: 0.25,
            MarketRegime.VOLATILE: 0.20,
            MarketRegime.BEAR_STRONG: 0.10,
            MarketRegime.BEAR_WEAK: 0.10,
        }

    def detect_regimes_for_bars(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20,
    ) -> List[MarketRegime]:
        """Detect regime for each bar in the series"""
        regimes = []
        for i in range(len(close)):
            if i < window:
                regimes.append(MarketRegime.UNKNOWN)
            else:
                regime = self.regime_detector.detect_regime(
                    high.iloc[: i + 1],
                    low.iloc[: i + 1],
                    close.iloc[: i + 1],
                    volume.iloc[: i + 1],
                    lookback=window,
                )
                regimes.append(regime)
        return regimes

    def compute_sampling_weights(
        self,
        regimes: List[MarketRegime],
        current_regime: MarketRegime = None,
        upsampling_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Compute sampling weights to balance regime representation.

        Args:
            regimes: List of detected regimes
            current_regime: Current regime (for focus sampling)
            upsampling_factor: Factor to upsample current regime

        Returns:
            Array of sampling weights
        """
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        weights = np.ones(len(regimes))

        for i, regime in enumerate(regimes):
            if regime == MarketRegime.UNKNOWN:
                weights[i] = 0
                continue

            target_pct = self.target_distribution.get(regime, 0.1)
            actual_pct = regime_counts.get(regime, 1) / len(regimes)

            if actual_pct > 0:
                weight = target_pct / actual_pct
            else:
                weight = 1.0

            if current_regime is not None and regime == current_regime:
                weight *= upsampling_factor

            weights[i] = weight

        weights = weights / weights.sum() * len(regimes)

        return weights

    def sample_regimes(
        self,
        n_samples: int,
        regimes: List[MarketRegime],
        weights: Optional[np.ndarray] = None,
        replacement: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample indices based on regime weights.

        Args:
            n_samples: Number of samples to draw
            regimes: List of regimes
            weights: Optional pre-computed weights
            replacement: Sample with replacement

        Returns:
            Tuple of (sampled_indices, sampled_regimes)
        """
        if weights is None:
            weights = self.compute_sampling_weights(regimes)

        indices = np.random.choice(
            len(regimes),
            size=n_samples,
            replace=replacement,
            p=weights / weights.sum(),
        )

        return indices, [regimes[i] for i in indices]


class RegimeAwareLoss:
    """
    Custom loss function that applies regime-based weighting.
    """

    def __init__(
        self,
        base_loss_fn,
        regime_weights: Optional[Dict[MarketRegime, float]] = None,
        focus_regime: Optional[MarketRegime] = None,
    ):
        self.base_loss_fn = base_loss_fn
        self.regime_weights = regime_weights or {
            MarketRegime.BULL_STRONG: 1.0,
            MarketRegime.BULL_WEAK: 1.2,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.VOLATILE: 1.5,
            MarketRegime.BEAR_STRONG: 1.0,
            MarketRegime.BEAR_WEAK: 1.2,
        }
        self.focus_regime = focus_regime

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regimes: List[MarketRegime],
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute regime-weighted loss.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            regimes: List of regimes for each sample
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Loss tensor
        """
        base_loss = self.base_loss_fn(predictions, targets, reduction="none")

        weights = torch.ones(len(regimes))
        for i, regime in enumerate(regimes):
            if regime in self.regime_weights:
                weights[i] = self.regime_weights[regime]
            if self.focus_regime is not None and regime == self.focus_regime:
                weights[i] *= 2.0

        weighted_loss = base_loss * weights.to(base_loss.device)

        if reduction == "mean":
            return weighted_loss.mean()
        elif reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


def create_regime_balanced_batches(
    features: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    regimes: List[MarketRegime],
    batch_size: int = 32,
    sampler: Optional[RegimeAwareSampler] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[MarketRegime]]:
    """
    Create regime-balanced batches for training.

    Args:
        features: Input features
        labels: Target labels
        confidences: Confidence scores
        regimes: Detected regimes
        batch_size: Batch size
        sampler: Regime-aware sampler

    Returns:
        Tuple of (batch_features, batch_labels, batch_confidences, batch_regimes)
    """
    if sampler is None:
        sampler = RegimeAwareSampler()

    weights = sampler.compute_sampling_weights(regimes)

    n_batches = len(features) // batch_size
    indices = np.random.choice(
        len(features),
        size=n_batches * batch_size,
        replace=False,
        p=weights / weights.sum(),
    )

    batch_features = features[indices].reshape(-1, batch_size, *features.shape[1:])
    batch_labels = labels[indices].reshape(-1, batch_size)
    batch_confidences = confidences[indices].reshape(-1, batch_size)
    batch_regimes = [regimes[i] for i in indices]
    batch_regimes = [
        batch_regimes[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)
    ]

    return batch_features, batch_labels, batch_confidences, batch_regimes


def compute_regime_performance(
    trades: pd.DataFrame,
    regimes: List[MarketRegime],
) -> Dict[MarketRegime, RegimePerformance]:
    """
    Compute performance metrics per regime.

    Args:
        trades: DataFrame of trades with 'pnl', 'regime' columns
        regimes: List of regimes for each bar

    Returns:
        Dict mapping regime to performance metrics
    """
    performance = {}

    for regime in MarketRegime:
        if regime == MarketRegime.UNKNOWN:
            continue

        regime_trades = trades[trades.get("regime", "unknown") == regime.value]

        if len(regime_trades) == 0:
            continue

        wins = (regime_trades["pnl"] > 0).sum()
        losses = (regime_trades["pnl"] < 0).sum()
        total = len(regime_trades)

        win_rate = wins / total if total > 0 else 0
        avg_return = regime_trades["pnl"].mean()
        profit_factor = (
            regime_trades[regime_trades["pnl"] > 0]["pnl"].sum()
            / abs(regime_trades[regime_trades["pnl"] < 0]["pnl"].sum())
            if losses > 0
            else float("inf")
        )

        performance[regime] = RegimePerformance(
            regime=regime,
            win_rate=win_rate,
            total_trades=total,
            winning_trades=wins,
            avg_confidence=regime_trades.get(
                "confidence", pd.Series([0.5] * total)
            ).mean(),
            avg_return=avg_return,
            profit_factor=profit_factor,
        )

    return performance


if __name__ == "__main__":
    np.random.seed(42)
    n_bars = 1000

    close = pd.Series(17500 * (1 + np.cumsum(np.random.randn(n_bars) * 0.001)))
    high = close + np.random.rand(n_bars) * 10
    low = close - np.random.rand(n_bars) * 10
    volume = pd.Series(np.random.randint(10000, 50000, n_bars))

    detector = RegimeDetector()
    regimes = detector.detect_regimes_for_bars(high, low, close, volume, window=20)

    print(f"Detected regimes:")
    for regime in MarketRegime:
        count = sum(1 for r in regimes if r == regime)
        print(f"  {regime.value}: {count} ({100 * count / len(regimes):.1f}%)")

    sampler = RegimeAwareSampler()
    weights = sampler.compute_sampling_weights(regimes)
    print(f"\nSampling weight stats:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
