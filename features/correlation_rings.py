"""
Hierarchical Correlation Rings Feature Generator

Computes day-by-day correlation matrices for MNQ and global assets,
then builds hierarchical "rings" for text encoder input.

Ring 1: US Equity Futures (MNQ, ES, NQ, RTY)
Ring 2: Global/Commodity Futures (CL, GC, ZB, 6E, FDAX)

Each ring has 14 days of correlation data (2-week rolling window),
enabling the text encoder to understand cross-asset relationships.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import sqrtm

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationRing:
    """Single correlation ring data structure"""

    name: str
    assets: List[str]
    daily_correlations: List[Dict[str, Any]] = field(default_factory=list)
    ring_correlation_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "assets": self.assets,
            "daily_correlations": self.daily_correlations,
            "matrix": self.ring_correlation_matrix.tolist()
            if self.ring_correlation_matrix is not None
            else None,
        }


@dataclass
class HierarchicalCorrelationData:
    """Complete hierarchical correlation data for text encoder"""

    ring1: Dict
    ring2: Dict
    cross_ring_correlation: Optional[np.ndarray] = None
    timestamp: str = ""
    text_description: str = ""
    detected_regime: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "ring1": self.ring1,
            "ring2": self.ring2,
            "cross_ring_correlation": self.cross_ring_correlation.tolist()
            if self.cross_ring_correlation is not None
            else None,
            "timestamp": self.timestamp,
            "text_description": self.text_description,
            "detected_regime": self.detected_regime,
        }


def compute_correlation_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Compute correlation matrix from returns DataFrame.

    Args:
        returns_df: DataFrame with returns for each asset (columns = assets)

    Returns:
        Correlation matrix (N x N)
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        logger.warning("Insufficient data for correlation matrix")
        return np.eye(len(returns_df.columns))

    corr_matrix = returns_df.corr().values

    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    return corr_matrix


def compute_spd_matrix(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Compute symmetric positive definite (SPD) matrix from correlation.

    Ensures numerical stability for downstream processing.

    Args:
        corr_matrix: Raw correlation matrix

    Returns:
        SPD matrix
    """
    spd = corr_matrix.copy()

    eigenvalues, eigenvectors = np.linalg.eigh(spd)

    min_eigenvalue = max(eigenvalues.min(), 1e-6)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    spd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return spd


def compute_tail_dependency(
    corr_matrix: np.ndarray, threshold: float = 0.7
) -> Dict[str, float]:
    """
    Compute tail dependency metrics using correlation matrix.

    Measures how likely extreme co-movements are between assets.

    Args:
        corr_matrix: Correlation matrix
        threshold: Threshold for "extreme" correlation

    Returns:
        Dict of tail dependency metrics
    """
    n = corr_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_values = corr_matrix[upper_tri_indices]

    metrics = {
        "tail_dependency_strength": float(np.mean(np.abs(upper_tri_values))),
        "num_tail_pairs": int(np.sum(np.abs(upper_tri_values) > threshold)),
        "max_tail_correlation": float(np.max(np.abs(upper_tri_values))),
        "tail_asymmetry": float(
            np.mean(np.maximum(upper_tri_values, 0))
            - np.mean(np.minimum(upper_tri_values, 0))
        ),
    }

    if metrics["tail_asymmetry"] > 0:
        metrics["tail_regime"] = "positive_momentum"
    elif metrics["tail_asymmetry"] < 0:
        metrics["tail_regime"] = "mean_reversion"
    else:
        metrics["tail_regime"] = "neutral"

    return metrics


def compute_regime_from_correlations(
    corr_matrix: np.ndarray, volatility: float = 0.01
) -> str:
    """
    Infer market regime from correlation structure.

    Args:
        corr_matrix: Correlation matrix
        volatility: Current volatility level

    Returns:
        Regime label
    """
    tail_metrics = compute_tail_dependency(corr_matrix)

    avg_corr = tail_metrics["tail_dependency_strength"]
    num_high_corr = tail_metrics["num_tail_pairs"]

    if avg_corr > 0.7 and num_high_corr > 3:
        if volatility > 0.015:
            return "crisis_regime"
        else:
            return "strong_trend"
    elif avg_corr > 0.4:
        if volatility > 0.02:
            return "volatile_trend"
        else:
            return "moderate_trend"
    elif avg_corr < 0.2:
        return "mean_reversion"
    else:
        return "sideways"


def compute_gaussian_copula_features(
    corr_matrix: np.ndarray, n_samples: int = 1000
) -> Dict[str, float]:
    """
    Generate synthetic samples using Gaussian copula for feature extraction.

    Args:
        corr_matrix: Correlation matrix
        n_samples: Number of samples to generate

    Returns:
        Dict of copula-based features
    """
    try:
        L = np.linalg.cholesky(corr_matrix)
        z = np.random.randn(n_samples, corr_matrix.shape[0])
        samples = z @ L.T

        uniform_samples = stats.norm.cdf(samples)

        features = {
            "copula_mean": float(np.mean(samples)),
            "copula_std": float(np.std(samples)),
            "copula_skew": float(stats.skew(samples.flatten())),
            "copula_kurtosis": float(stats.kurtosis(samples.flatten())),
            "extreme_co_movements": float(np.mean(np.abs(samples) > 2) / 0.05),
        }

        return features
    except np.linalg.LinAlgError:
        return {
            "copula_mean": 0.0,
            "copula_std": 1.0,
            "copula_skew": 0.0,
            "copula_kurtosis": 0.0,
            "extreme_co_movements": 1.0,
        }


def compute_correlation_features(
    corr_matrix: np.ndarray, assets: List[str]
) -> Dict[str, Any]:
    """
    Extract features from a correlation matrix.

    Args:
        corr_matrix: N x N correlation matrix
        assets: List of asset names

    Returns:
        Dict of correlation features
    """
    n = len(assets)

    features = {
        "n_assets": n,
        "mean_correlation": 0.0,
        "std_correlation": 0.0,
        "max_correlation": 0.0,
        "min_correlation": 0.0,
        "num_high_corr_pairs": 0,
        "matrix_frobenius_norm": 0.0,
        "condition_number": 0.0,
        "eigenvalue_spread": 0.0,
        "network_density": 0.0,
        "avg_abs_correlation": 0.0,
    }

    if n < 2:
        return features

    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    if len(upper_tri) > 0:
        features["mean_correlation"] = float(np.mean(upper_tri))
        features["std_correlation"] = float(np.std(upper_tri))
        features["max_correlation"] = float(np.max(upper_tri))
        features["min_correlation"] = float(np.min(upper_tri))
        features["num_high_corr_pairs"] = int(np.sum(np.abs(upper_tri) > 0.7))
        features["avg_abs_correlation"] = float(np.mean(np.abs(upper_tri)))

    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    features["eigenvalue_spread"] = float(np.max(eigenvalues) - np.min(eigenvalues))
    features["matrix_frobenius_norm"] = float(np.linalg.norm(corr_matrix, "fro"))
    features["condition_number"] = float(np.linalg.cond(corr_matrix))
    features["network_density"] = float(np.sum(np.abs(corr_matrix) > 0.5) / (n * n))

    tail_metrics = compute_tail_dependency(corr_matrix)
    features.update(tail_metrics)

    return features


def matrix_power_sqrt(corr_matrix: np.ndarray, power: float = 0.5) -> np.ndarray:
    """
    Compute matrix power (e.g., square root) of correlation matrix.

    Used in ring-to-ring propagation.

    Args:
        corr_matrix: SPD correlation matrix
        power: Power to raise eigenvalues (0.5 = sqrt)

    Returns:
        Matrix raised to specified power
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    eigenvalues = np.maximum(eigenvalues, 1e-8)
    eigenvalues_powered = np.power(eigenvalues, power)

    result = eigenvectors @ np.diag(eigenvalues_powered) @ eigenvectors.T

    return result


def normalize_correlation_for_text(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize correlation matrix to [0, 1] range for text encoding.

    Args:
        corr_matrix: Raw correlation matrix

    Returns:
        Normalized matrix
    """
    normalized = (corr_matrix + 1) / 2
    normalized = np.clip(normalized, 0, 1)
    return normalized


class CorrelationRingGenerator:
    """
    Generates hierarchical correlation rings for MNQ and global assets.
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.ring1_assets = ["MNQ", "ES", "NQ", "RTY"]
        self.ring2_assets = ["CL", "GC", "ZB", "6E", "FDAX"]
        self.history_days = self.config["data"].get("history_days", 14)

    def compute_daily_returns(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute daily returns for each asset.

        Args:
            price_data: Dict mapping symbol -> OHLCV DataFrame

        Returns:
            Dict mapping symbol -> returns DataFrame
        """
        returns_data = {}

        for symbol, df in price_data.items():
            if "close" not in df.columns:
                logger.warning(f"No close column for {symbol}")
                continue

            close = df["close"].dropna()
            returns = close.pct_change().dropna()

            returns_data[symbol] = returns

        return returns_data

    def build_daily_correlation(
        self, returns_data: Dict[str, pd.DataFrame], assets: List[str], date: datetime
    ) -> Dict[str, Any]:
        """
        Build correlation matrix for a specific date.

        Args:
            returns_data: Dict of returns DataFrames
            assets: List of asset symbols
            date: Target date

        Returns:
            Dict containing correlation data for the date
        """
        aligned_returns = []

        for symbol in assets:
            if symbol in returns_data:
                asset_returns = returns_data[symbol]
                asset_returns.index = pd.to_datetime(asset_returns.index)
                date_returns = asset_returns[asset_returns.index.date == date.date()]

                if len(date_returns) > 0:
                    aligned_returns.append(date_returns.values)

        if len(aligned_returns) < 2:
            return {
                "date": date.isoformat(),
                "correlation_matrix": np.eye(len(assets)).tolist(),
                "features": compute_correlation_features(np.eye(len(assets)), assets),
                "status": "insufficient_data",
            }

        min_len = min(len(r) for r in aligned_returns)
        aligned_returns = [r[:min_len] for r in aligned_returns]

        returns_matrix = np.column_stack(aligned_returns)
        returns_df = pd.DataFrame(returns_matrix, columns=assets)

        corr_matrix = compute_correlation_matrix(returns_df)
        spd_matrix = compute_spd_matrix(corr_matrix)
        features = compute_correlation_features(corr_matrix, assets)

        return {
            "date": date.isoformat(),
            "correlation_matrix": normalize_correlation_for_text(corr_matrix).tolist(),
            "spd_matrix": spd_matrix.tolist(),
            "features": features,
            "status": "success",
        }

    def build_ring_for_period(
        self,
        returns_data: Dict[str, pd.DataFrame],
        assets: List[str],
        start_date: datetime,
        num_days: int = 14,
    ) -> CorrelationRing:
        """
        Build correlation ring for a period of days.

        Args:
            returns_data: Dict of returns DataFrames
            assets: List of asset symbols
            start_date: End date of the period
            num_days: Number of days in the window

        Returns:
            CorrelationRing with daily correlation data
        """
        ring = CorrelationRing(
            name="_".join(assets), assets=assets, daily_correlations=[]
        )

        for i in range(num_days):
            date = start_date - timedelta(days=i)
            daily_corr = self.build_daily_correlation(returns_data, assets, date)
            ring.daily_correlations.append(daily_corr)

        ring.daily_correlations.reverse()

        if len(ring.daily_correlations) > 0:
            matrices = [
                np.array(d["correlation_matrix"])
                for d in ring.daily_correlations
                if d["status"] == "success"
            ]
            if matrices:
                avg_matrix = np.mean(matrices, axis=0)
                ring.ring_correlation_matrix = compute_spd_matrix(avg_matrix)

        return ring

    def propagate_ring_matrix(
        self, ring1_matrix: np.ndarray, ring2_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Propagate correlation information between rings.

        Matrix multiplication: R1 × R2 captures cross-ring relationships.

        Args:
            ring1_matrix: Correlation matrix for Ring 1
            ring2_matrix: Correlation matrix for Ring 2

        Returns:
            Cross-ring correlation matrix
        """
        cross_corr = ring1_matrix @ ring2_matrix

        cross_corr = normalize_correlation_for_text(cross_corr)

        return cross_corr

    def generate_text_description(
        self,
        ring1: CorrelationRing,
        ring2: CorrelationRing,
        current_regime: str = "unknown",
    ) -> str:
        """
        Generate natural language description of correlation state.

        Used as input to the text encoder.

        Args:
            ring1: Ring 1 correlation data
            ring2: Ring 2 correlation data
            current_regime: Detected market regime

        Returns:
            Text description of market correlations
        """
        lines = []

        lines.append("Market Correlation State Report.")
        lines.append(f"Analysis Period: {self.history_days} trading days.")
        lines.append(f"Current Regime: {current_regime}.")

        ring1_features = (
            ring1.daily_correlations[-1]["features"] if ring1.daily_correlations else {}
        )
        ring2_features = (
            ring2.daily_correlations[-1]["features"] if ring2.daily_correlations else {}
        )

        avg_corr_r1 = ring1_features.get("mean_correlation", 0)
        avg_corr_r2 = ring2_features.get("mean_correlation", 0)
        tail_r1 = ring1_features.get("tail_regime", "neutral")
        tail_r2 = ring2_features.get("tail_regime", "neutral")

        lines.append(
            f"US Equity Futures (MNQ, ES, NQ, RTY): Mean correlation {avg_corr_r1:.2f}, tail regime {tail_r1}."
        )

        if avg_corr_r1 > 0.8:
            lines.append("Strong correlation regime among US equity futures.")
        elif avg_corr_r1 > 0.5:
            lines.append("Moderate correlation among US equity futures.")
        else:
            lines.append("Weak correlation among US equity futures.")

        lines.append(
            f"Global/Commodity Futures (CL, GC, ZB, 6E, FDAX): Mean correlation {avg_corr_r2:.2f}, tail regime {tail_r2}."
        )

        if avg_corr_r2 > 0.6:
            lines.append("Commodity markets show correlated movement.")
        elif avg_corr_r2 > 0.3:
            lines.append("Mixed correlation in global markets.")
        else:
            lines.append("Diverse movement across asset classes.")

        if current_regime in ["crisis_regime", "strong_trend"]:
            lines.append(
                "Market showing directional conviction - trend following signals favored."
            )
        elif current_regime == "mean_reversion":
            lines.append(
                "Market showing non-correlated behavior - mean reversion signals favored."
            )
        else:
            lines.append(
                "Market in neutral regime - balanced signal approach recommended."
            )

        return " ".join(lines)


class HierarchicalCorrelationEngine:
    """
    Complete engine for generating hierarchical correlation data.
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        self.generator = CorrelationRingGenerator(config_path)

    def process(
        self,
        price_data: Dict[str, pd.DataFrame],
        reference_date: Optional[datetime] = None,
        current_volatility: float = 0.01,
    ) -> HierarchicalCorrelationData:
        """
        Process price data and generate hierarchical correlation rings.

        Args:
            price_data: Dict mapping symbol -> OHLCV DataFrame
            reference_date: Reference date (default: now)
            current_volatility: Current market volatility level

        Returns:
            HierarchicalCorrelationData with all ring information
        """
        if reference_date is None:
            reference_date = datetime.now()

        returns_data = self.generator.compute_daily_returns(price_data)

        ring1 = self.generator.build_ring_for_period(
            returns_data,
            self.generator.ring1_assets,
            reference_date,
            self.generator.history_days,
        )

        ring2 = self.generator.build_ring_for_period(
            returns_data,
            self.generator.ring2_assets,
            reference_date,
            self.generator.history_days,
        )

        ring1_features = (
            ring1.daily_correlations[-1]["features"] if ring1.daily_correlations else {}
        )
        ring2_features = (
            ring2.daily_correlations[-1]["features"] if ring2.daily_correlations else {}
        )

        corr_matrix_r1 = (
            np.array(ring1_features.get("correlation_matrix", np.eye(4)))
            if "correlation_matrix" in ring1_features
            else np.eye(4)
        )
        corr_matrix_r2 = (
            np.array(ring2_features.get("correlation_matrix", np.eye(5)))
            if "correlation_matrix" in ring2_features
            else np.eye(5)
        )

        regime = compute_regime_from_correlations(corr_matrix_r1, current_volatility)

        cross_ring = None
        if (
            ring1.ring_correlation_matrix is not None
            and ring2.ring_correlation_matrix is not None
        ):
            cross_ring = self.generator.propagate_ring_matrix(
                ring1.ring_correlation_matrix, ring2.ring_correlation_matrix
            )

        text_description = self.generator.generate_text_description(
            ring1, ring2, regime
        )

        data = HierarchicalCorrelationData(
            ring1=ring1.to_dict(),
            ring2=ring2.to_dict(),
            cross_ring_correlation=cross_ring,
            timestamp=reference_date.isoformat(),
        )

        data.text_description = text_description
        data.detected_regime = regime

        logger.info(f"Generated hierarchical correlation data - Regime: {regime}")
        return data


def flatten_correlation_for_bert(data: HierarchicalCorrelationData) -> np.ndarray:
    """
    Flatten correlation data into a vector for BERT encoding.

    Args:
        data: Hierarchical correlation data

    Returns:
        Flattened correlation vector
    """
    vectors = []

    r1_matrix = np.array(data.ring1["correlation_matrix"])
    r2_matrix = np.array(data.ring2["correlation_matrix"])

    r1_flat = r1_matrix.flatten()
    r2_flat = r2_matrix.flatten()

    r1_features = data.ring1.get("features", {})
    r2_features = data.ring2.get("features", {})

    feature_vec = [
        r1_features.get("mean_correlation", 0),
        r1_features.get("std_correlation", 0),
        r1_features.get("max_correlation", 0),
        r1_features.get("min_correlation", 0),
        r1_features.get("tail_dependency_strength", 0),
        r1_features.get("eigenvalue_spread", 0),
        r1_features.get("network_density", 0),
        r2_features.get("mean_correlation", 0),
        r2_features.get("std_correlation", 0),
        r2_features.get("max_correlation", 0),
        r2_features.get("min_correlation", 0),
        r2_features.get("tail_dependency_strength", 0),
        r2_features.get("eigenvalue_spread", 0),
        r2_features.get("network_density", 0),
        r1_features.get("condition_number", 1),
        r2_features.get("condition_number", 1),
    ]

    regime_map = {
        "strong_trend": [1, 0, 0, 0, 0],
        "moderate_trend": [0, 1, 0, 0, 0],
        "volatile_trend": [0, 0, 1, 0, 0],
        "mean_reversion": [0, 0, 0, 1, 0],
        "sideways": [0, 0, 0, 0, 1],
        "crisis_regime": [1, 0, 0, 0, 0],
        "unknown": [0, 0, 0, 0, 0],
    }
    regime_vec = regime_map.get(data.detected_regime, regime_map["unknown"])

    vectors.extend(r1_flat[:36])
    vectors.extend(r2_flat[:36])
    vectors.extend(feature_vec)
    vectors.extend(regime_vec)

    result = np.array(vectors[:128], dtype=np.float32)

    if len(result) < 128:
        result = np.pad(result, (0, 128 - len(result)))

    return result


if __name__ == "__main__":
    engine = HierarchicalCorrelationEngine()

    import numpy as np

    dates = pd.date_range("2024-01-01", periods=14, freq="D")

    sample_data = {}
    for symbol in ["MNQ", "ES", "NQ", "RTY", "CL", "GC", "ZB", "6E", "FDAX"]:
        base_price = {
            "MNQ": 17000,
            "ES": 5000,
            "NQ": 17500,
            "RTY": 2000,
            "CL": 75,
            "GC": 2000,
            "ZB": 120,
            "6E": 1.08,
            "FDAX": 17000,
        }.get(symbol, 100)

        sample_data[symbol] = pd.DataFrame(
            {"close": base_price + np.cumsum(np.random.randn(14) * base_price * 0.01)},
            index=dates,
        )

    result = engine.process(sample_data)

    print(f"Ring 1 assets: {result.ring1['assets']}")
    print(
        f"Ring 1 mean correlation: {result.ring1['features']['mean_correlation']:.3f}"
    )
    print(f"Ring 2 assets: {result.ring2['assets']}")
    print(
        f"Ring 2 mean correlation: {result.ring2['features']['mean_correlation']:.3f}"
    )
    print(f"\nText description:\n{result.text_description}")
