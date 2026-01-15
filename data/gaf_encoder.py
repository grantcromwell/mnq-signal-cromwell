"""
GAF (Gramian Angular Field) Encoder for MNQ Price Action

Converts 1D OHLC time series data into 2D images while preserving
temporal relationships and enabling vision transformer processing.

Supports:
- GASF (Gramian Angular Summation Field)
- GADF (Gramian Angular Difference Field)
- Multi-channel encoding (OHC as RGB)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GAFConfig:
    """GAF encoding configuration"""

    resolution: int = 64  # 64x64 per day
    days_main: int = 3  # 3-day GAF for main encoding
    days_week: int = 5  # Week GAF for context
    channels: int = 3  # OHC (Open, High, Close)
    gaf_type: str = "gasf"  # gasf or gadf
    normalize: bool = True
    rescale: bool = True
    value_range: Tuple[float, float] = (0, 1)


class GAFEncoder:
    """
    Gramian Angular Field encoder for trading data.
    Converts OHLC time series to 2D images for vision models.
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["gaf"]

        self.resolution = self.config["resolution"]
        self.days_main = self.config.get("days_3", 3)
        self.days_week = self.config["days_week"]
        self.channels = self.config["channels"]
        self.gaf_type = self.config["gaf_type"]
        self.normalize = self.config.get("normalize", True)
        self.rescale = self.config.get("rescale", True)

    def normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Normalize prices to [0, 1] range using min-max scaling.

        Args:
            prices: Array of price values

        Returns:
            Normalized prices in [0, 1]
        """
        if self.rescale:
            min_val = prices.min()
            max_val = prices.max()
            if max_val - min_val > 0:
                return (prices - min_val) / (max_val - min_val)
        return prices

    def to_polar_coordinates(
        self, normalized_prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized prices to polar coordinates.

        The time series is transformed to angular coordinates where:
        - φ = arccos(price_normalized)  (angle)
        - r = t / T  (radius, normalized time)

        Args:
            normalized_prices: Normalized price values [0, 1]

        Returns:
            Tuple of (phi, r) arrays
        """
        phi = np.arccos(np.clip(normalized_prices, -1, 1))
        t = np.arange(len(normalized_prices))
        r = t / (len(t) - 1) if len(t) > 1 else np.zeros_like(t)
        return phi, r

    def compute_gasf(self, phi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute Gramian Angular Summation Field.

        GASF(i, j) = cos(φ_i + φ_j)

        This captures the temporal relationship between all pairs
        of points in the time series.

        Args:
            phi: Angular coordinates
            r: Radial coordinates

        Returns:
            GASF matrix
        """
        cos_sum = np.cos(np.add.outer(phi, phi))
        return cos_sum

    def compute_gadf(self, phi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Compute Gramian Angular Difference Field.

        GADF(i, j) = sin(φ_i - φ_j)

        Captures different temporal relationships than GASF.

        Args:
            phi: Angular coordinates
            r: Radial coordinates

        Returns:
            GADF matrix
        """
        sin_diff = np.sin(np.subtract.outer(phi, phi))
        return sin_diff

    def compute_gaf(self, phi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Compute GAF based on configured type"""
        if self.gaf_type == "gasf":
            return self.compute_gasf(phi, r)
        elif self.gaf_type == "gadf":
            return self.compute_gadf(phi, r)
        else:
            raise ValueError(f"Unknown GAF type: {self.gaf_type}")

    def encode_single_channel(
        self, prices: np.ndarray, target_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode a single price series to GAF image.

        Args:
            prices: Array of price values
            target_size: Target output size (default: config resolution)

        Returns:
            GAF image as numpy array (H, W)
        """
        if target_size is None:
            target_size = self.resolution

        if len(prices) < 2:
            raise ValueError("Need at least 2 price points for GAF")

        normalized = self.normalize_prices(prices)
        phi, r = self.to_polar_coordinates(normalized)
        gaf = self.compute_gaf(phi, r)

        if gaf.shape[0] != target_size:
            from scipy.ndimage import zoom

            zoom_factors = (target_size / gaf.shape[0], target_size / gaf.shape[0])
            gaf = zoom(gaf, zoom_factors, order=1)

        return gaf

    def encode_ohlc_to_image(
        self, df: pd.DataFrame, target_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode OHLC data to multi-channel GAF image.

        Creates:
        - Channel 0: Open prices (O)
        - Channel 1: High prices (H)
        - Channel 2: Close prices (C)

        Args:
            df: DataFrame with 'open', 'high', 'close' columns
            target_size: Target image size (H, W)

        Returns:
            GAF image as numpy array (C, H, W)
        """
        if target_size is None:
            target_size = self.resolution

        if self.channels == 3:
            price_cols = ["open", "high", "close"]
        elif self.channels == 4:
            price_cols = ["open", "high", "low", "close"]
        else:
            price_cols = ["close"]

        channels = []
        for col in price_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

            prices = df[col].values.astype(float)
            gaf = self.encode_single_channel(prices, target_size)
            channels.append(gaf)

        image = np.stack(channels, axis=0)

        return image

    def encode_day_range(
        self, df: pd.DataFrame, num_days: int, interval: str = "15m"
    ) -> np.ndarray:
        """
        Encode price data for a specific number of trading days.

        Args:
            df: Full DataFrame with OHLC data
            num_days: Number of days to encode
            interval: Bar interval

        Returns:
            Combined GAF image for the period
        """
        if len(df) == 0:
            raise ValueError("Empty DataFrame")

        bars_per_day = {
            "1m": 390,  # 6.5 hours * 60 (RTH)
            "5m": 78,
            "15m": 26,
            "1h": 7,
        }

        bars_per_day = bars_per_day.get(interval, 26)
        required_bars = num_days * bars_per_day

        if len(df) < required_bars:
            logger.warning(f"Insufficient data: {len(df)} < {required_bars}")
            df_subset = df
        else:
            df_subset = df.tail(required_bars)

        image = self.encode_ohlc_to_image(df_subset, self.resolution)

        return image

    def create_3day_gaf(self, df: pd.DataFrame) -> np.ndarray:
        """Create 3-day GAF for main encoding"""
        return self.encode_day_range(df, self.days_main)

    def create_week_gaf(self, df: pd.DataFrame) -> np.ndarray:
        """Create week GAF for context"""
        return self.encode_day_range(df, self.days_week)


def resample_to_daily_bars(
    df: pd.DataFrame, value_column: str = "close"
) -> pd.DataFrame:
    """
    Resample intraday bars to daily bars for week encoding.

    Args:
        df: Intraday DataFrame with OHLCV data
        value_column: Column to use for resampling

    Returns:
        Daily bar DataFrame
    """
    if df.empty:
        return df

    daily = (
        df.resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    return daily


class GAFDatasetGenerator:
    """
    Generates training datasets from GAF-encoded price images.
    """

    def __init__(self, gaf_encoder: GAFEncoder):
        self.encoder = gaf_encoder

    def generate_training_sample(
        self,
        mnq_df: pd.DataFrame,
        week_df: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single training sample with both GAF images.

        Args:
            mnq_df: 3-day OHLCV data
            week_df: Weekly context OHLCV data (if None, derived from mnq_df)
            timestamp: Sample timestamp

        Returns:
            Dict containing:
            - 'image_3day': (3, 64, 64) numpy array
            - 'image_week': (3, 64, 64) numpy array
            - 'timestamp': datetime
        """
        image_3day = self.encoder.create_3day_gaf(mnq_df)

        if week_df is None:
            week_df = resample_to_daily_bars(mnq_df)

        image_week = self.encoder.create_week_gaf(week_df)

        return {
            "image_3day": image_3day,
            "image_week": image_week,
            "timestamp": timestamp or datetime.now(),
        }

    def batch_generate(
        self, df: pd.DataFrame, window_days: int = 3, step_bars: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple samples from a DataFrame.

        Args:
            df: Full OHLCV DataFrame
            window_days: Days per sample
            step_bars: Bars between consecutive samples

        Returns:
            List of sample dictionaries
        """
        bars_per_day = 26  # 15-min bars
        window_bars = window_days * bars_per_day

        samples = []
        for i in range(0, len(df) - window_bars, step_bars):
            window_df = df.iloc[i : i + window_bars]
            sample = self.generate_training_sample(window_df)
            samples.append(sample)

        logger.info(f"Generated {len(samples)} training samples")
        return samples


def visualize_gaf(gaf_image: np.ndarray, title: str = "GAF Image"):
    """Display GAF image (for debugging)"""
    import matplotlib.pyplot as plt

    if gaf_image.shape[0] == 3:
        plt.imshow(np.transpose(gaf_image, (1, 2, 0)))
    else:
        plt.imshow(gaf_image, cmap="viridis")

    plt.title(title)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    encoder = GAFEncoder()

    sample_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "open": np.cumsum(np.random.randn(100)) + 17000,
            "high": np.cumsum(np.random.randn(100)) + 17005,
            "low": np.cumsum(np.random.randn(100)) + 16995,
            "close": np.cumsum(np.random.randn(100)) + 17000,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )

    gaf_3day = encoder.create_3day_gaf(sample_data)
    print(f"3-day GAF shape: {gaf_3day.shape}")

    daily = resample_to_daily_bars(sample_data)
    gaf_week = encoder.create_week_gaf(daily)
    print(f"Week GAF shape: {gaf_week.shape}")
