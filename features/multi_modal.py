"""
Multi-Modal Feature Engineering for Enhanced MNQ Trading

Combines:
- GAF Images (Vision)
- Correlation Rings (Text)
- News Sentiment (Alpha Vantage)
- Technical Indicators
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiModalFeatures:
    """Container for all multi-modal features"""

    gaf_3day: np.ndarray
    gaf_week: np.ndarray
    correlation_vec: np.ndarray
    sentiment_vec: np.ndarray
    technical_vec: np.ndarray
    regime: str
    combined: np.ndarray


class MultiModalFeatureEngine:
    """
    Feature engineering combining all data modalities.
    """

    def __init__(self):
        self.regime_weights = {
            "bull_strong": np.array([1.0, 0.8, 0.5, 0.3]),
            "bull_weak": np.array([0.8, 1.0, 0.6, 0.4]),
            "sideways": np.array([0.5, 0.5, 0.8, 0.6]),
            "volatile": np.array([0.3, 0.4, 0.4, 0.8]),
            "bear_weak": np.array([0.2, 0.3, 0.5, 1.0]),
            "bear_strong": np.array([0.1, 0.2, 0.4, 1.0]),
        }

    def create_combined_features(
        self,
        correlation_vec: np.ndarray,
        sentiment_vec: np.ndarray,
        technical_vec: np.ndarray,
        regime: str = "unknown",
    ) -> np.ndarray:
        """
        Fuse all feature vectors into combined embedding.
        """
        weights = self.regime_weights.get(regime, self.regime_weights["sideways"])

        correlation = correlation_vec[:64] * weights[0]
        sentiment = sentiment_vec[:25] * weights[1]
        technical = technical_vec[:30] * weights[2]

        combined = np.concatenate([correlation, sentiment, technical])

        if len(combined) < 120:
            combined = np.pad(combined, (0, 120 - len(combined)))

        return combined[:120]

    def normalize_features(
        self,
        features: np.ndarray,
        mean: np.ndarray = None,
        std: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features with Z-score normalization.
        """
        if mean is None:
            mean = np.mean(features, axis=0)
        if std is None:
            std = np.std(features, axis=0)
            std[std == 0] = 1

        normalized = (features - mean) / std
        return normalized, mean, std


def create_regime_from_features(
    sentiment_vec: np.ndarray,
    technical_vec: np.ndarray,
) -> str:
    """
    Infer regime from feature vectors.
    """
    sentiment_score = sentiment_vec[0] if len(sentiment_vec) > 0 else 0

    price_trend = technical_vec[0] if len(technical_vec) > 0 else 0
    rsi = technical_vec[2] if len(technical_vec) > 2 else 0.5

    trend_score = sentiment_score * 0.4 + price_trend * 0.3 + (rsi - 0.5) * 0.3

    if trend_score > 0.3:
        if abs(trend_score) > 0.6:
            return "bull_strong"
        else:
            return "bull_weak"
    elif trend_score < -0.3:
        if abs(trend_score) > 0.6:
            return "bear_strong"
        else:
            return "bear_weak"
    else:
        volatility = technical_vec[5] if len(technical_vec) > 5 else 0
        if volatility > 0.5:
            return "volatile"
        else:
            return "sideways"


def create_regime_embedding(regime: str, embed_dim: int = 24) -> np.ndarray:
    """
    Create one-hot embedding for regime.
    """
    regime_map = {
        "bull_strong": 0,
        "bull_weak": 1,
        "sideways": 2,
        "volatile": 3,
        "bear_weak": 4,
        "bear_strong": 5,
        "unknown": 2,
    }

    regime_idx = regime_map.get(regime, 2)
    embedding = np.zeros(embed_dim)
    embedding[regime_idx * 4 : (regime_idx + 1) * 4] = 1
    embedding[regime_idx] = 1.0

    return embedding


if __name__ == "__main__":
    engine = MultiModalFeatureEngine()

    correlation = np.random.randn(64)
    sentiment = np.random.randn(25)
    technical = np.random.randn(30)

    combined = engine.create_combined_features(
        correlation, sentiment, technical, regime="bull_weak"
    )
    print(f"Combined features shape: {combined.shape}")

    regime = create_regime_from_features(sentiment, technical)
    print(f"Detected regime: {regime}")

    reg_emb = create_regime_embedding(regime)
    print(f"Regime embedding shape: {reg_emb.shape}")
