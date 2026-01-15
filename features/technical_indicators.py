"""
Technical Indicators Feature Generator for MNQ Trading Signal System

Adds traditional technical analysis indicators as additional features
to complement GAF images and correlation data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TechnicalFeatures:
    """Container for technical indicator features"""

    trend_features: np.ndarray
    momentum_features: np.ndarray
    volatility_features: np.ndarray
    volume_features: np.ndarray
    pattern_features: np.ndarray
    combined_features: np.ndarray


class TechnicalIndicatorGenerator:
    """
    Generates technical indicator features for trading signal system.

    Features include:
    - Trend: SMA, EMA, ADX, Ichimoku
    - Momentum: RSI, MACD, Stochastic, ROC
    - Volatility: ATR, Bollinger Bands, Keltner Channels
    - Volume: OBV, VWAP, Volume MA
    - Patterns: Doji, Hammer, Engulfing detection
    """

    def __init__(
        self,
        sma_periods: List[int] = [10, 20, 50, 100, 200],
        ema_periods: List[int] = [12, 26],
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stochastic_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_ma_periods: List[int] = [10, 20, 50],
    ):
        self.sma_periods = sma_periods
        self.ema_periods = ema_periods
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stochastic_period = stochastic_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_ma_periods = volume_ma_periods

    def compute_sma(self, close: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return close.rolling(window=period, min_periods=1).mean()

    def compute_ema(self, close: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return close.ewm(span=period, adjust=False).mean()

    def compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def compute_macd(
        self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = self.compute_ema(close, fast)
        ema_slow = self.compute_ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.compute_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def compute_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator %K and %D"""
        lowest_low = low.rolling(window=period, min_periods=1).min()
        highest_high = high.rolling(window=period, min_periods=1).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(window=3, min_periods=1).mean()
        return k.fillna(50), d.fillna(50)

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    def compute_bollinger_bands(
        self, close: pd.Series, period: int = 20, std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = close.rolling(window=period, min_periods=1).mean()
        std_dev = close.rolling(window=period, min_periods=1).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        bandwidth = (upper - lower) / sma
        percent = (close - lower) / (upper - lower).replace(0, np.nan)
        return upper, lower, percent

    def compute_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    def compute_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        cumulative_tpv = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tpv / cumulative_vol.replace(0, np.nan)
        return vwap

    def compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = pd.concat(
            [
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1)),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        plus_di = 100 * (plus_dm / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm / atr.replace(0, np.nan))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period, min_periods=1).mean()
        return adx.fillna(25)

    def compute_ichimoku(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Ichimoku Cloud components"""
        conv_period = 9
        base_period = 26
        leading_span_b_period = 52
        lagging_span_period = 26

        conversion = (
            high.rolling(window=conv_period, min_periods=1).max()
            + low.rolling(window=conv_period, min_periods=1).min()
        ) / 2
        base = (
            high.rolling(window=base_period, min_periods=1).max()
            + low.rolling(window=base_period, min_periods=1).min()
        ) / 2
        leading_span_a = (conversion + base) / 2
        leading_span_b = (
            high.rolling(window=leading_span_b_period, min_periods=1).max()
            + low.rolling(window=leading_span_b_period, min_periods=1).min()
        ) / 2
        lagging_span = close.shift(-lagging_span_period)

        return {
            "conversion": conversion,
            "base": base,
            "leading_span_a": leading_span_a,
            "leading_span_b": leading_span_b,
            "lagging_span": lagging_span,
        }

    def detect_candlestick_patterns(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> Dict[str, pd.Series]:
        """Detect common candlestick patterns"""
        body_size = abs(close - open_)
        upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low
        body_range = high - low

        patterns = {}

        doji = (body_size / body_range.replace(0, np.nan)) < 0.1
        patterns["doji"] = doji.astype(float)

        hammer = (
            (lower_shadow > 2 * body_size)
            & (upper_shadow < body_size * 0.3)
            & (close > open_)
        )
        patterns["hammer"] = hammer.astype(float)

        inverted_hammer = (
            (upper_shadow > 2 * body_size)
            & (lower_shadow < body_size * 0.3)
            & (close > open_)
        )
        patterns["inverted_hammer"] = inverted_hammer.astype(float)

        bullish_engulfing = (
            (close > open_)
            & (close.shift(1) < open_.shift(1))
            & (close > open_.shift(1))
            & (open_ < close.shift(1))
        )
        patterns["bullish_engulfing"] = bullish_engulfing.astype(float)

        bearish_engulfing = (
            (close < open_)
            & (close.shift(1) > open_.shift(1))
            & (close < open_.shift(1))
            & (open_ > close.shift(1))
        )
        patterns["bearish_engulfing"] = bearish_engulfing.astype(float)

        three_white_soldiers = (
            (close > open_)
            & (close.shift(1) > open_.shift(1))
            & (close.shift(2) > open_.shift(2))
            & (close > close.shift(1))
            & (close.shift(1) > close.shift(2))
        )
        patterns["three_white_soldiers"] = three_white_soldiers.astype(float)

        three_black_crows = (
            (close < open_)
            & (close.shift(1) < open_.shift(1))
            & (close.shift(2) < open_.shift(2))
            & (close < close.shift(1))
            & (close.shift(1) < close.shift(2))
        )
        patterns["three_black_crows"] = three_black_crows.astype(float)

        return patterns

    def compute_all_features(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> TechnicalFeatures:
        """Compute all technical indicator features"""

        trend_features = []
        for period in self.sma_periods:
            sma = self.compute_sma(close, period)
            trend_features.append((close - sma) / sma.replace(0, np.nan))
        for period in self.ema_periods:
            ema = self.compute_ema(close, period)
            trend_features.append((close - ema) / ema.replace(0, np.nan))

        adx = self.compute_adx(high, low, close)
        trend_features.append(adx / 100)

        ichimoku = self.compute_ichimoku(high, low, close)
        trend_features.append(
            (close - ichimoku["conversion"]) / close.replace(0, np.nan)
        )
        trend_features.append((close - ichimoku["base"]) / close.replace(0, np.nan))

        trend_features = np.column_stack(trend_features)

        momentum_features = []
        rsi = self.compute_rsi(close, self.rsi_period)
        momentum_features.append(rsi / 100)

        macd, macd_signal_line, macd_hist = self.compute_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        momentum_features.append(macd / close.replace(0, np.nan))
        momentum_features.append(macd_hist / close.replace(0, np.nan))

        stoch_k, stoch_d = self.compute_stochastic(
            high, low, close, self.stochastic_period
        )
        momentum_features.append(stoch_k / 100)
        momentum_features.append(stoch_d / 100)

        roc = (close - close.shift(10)) / close.shift(10).replace(0, np.nan)
        momentum_features.append(roc.fillna(0))

        momentum_features = np.column_stack(momentum_features)

        volatility_features = []
        atr = self.compute_atr(high, low, close, self.atr_period)
        volatility_features.append(atr / close.replace(0, np.nan))

        bb_upper, bb_lower, bb_percent = self.compute_bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        volatility_features.append(bb_percent.fillna(0.5))
        volatility_features.append(
            (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        )

        volatility_features = np.column_stack(volatility_features)

        volume_features = []
        for period in self.volume_ma_periods:
            vol_ma = volume.rolling(window=period, min_periods=1).mean()
            volume_features.append((volume - vol_ma) / vol_ma.replace(0, np.nan))

        obv = self.compute_obv(close, volume)
        obv_normalized = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(
            0, np.nan
        )
        volume_features.append(obv_normalized.fillna(0))

        vwap = self.compute_vwap(high, low, close, volume)
        volume_features.append((close - vwap) / vwap.replace(0, np.nan))

        volume_features = np.column_stack(volume_features)

        pattern_features = []
        patterns = self.detect_candlestick_patterns(open_, high, low, close)
        for pattern_name, pattern_values in patterns.items():
            pattern_features.append(pattern_values.fillna(0))

        pattern_features = np.column_stack(pattern_features)

        combined_features = np.hstack(
            [
                trend_features,
                momentum_features,
                volatility_features,
                volume_features,
                pattern_features,
            ]
        )

        return TechnicalFeatures(
            trend_features=trend_features,
            momentum_features=momentum_features,
            volatility_features=volatility_features,
            volume_features=volume_features,
            pattern_features=pattern_features,
            combined_features=combined_features,
        )

    def get_feature_names(self) -> List[str]:
        """Get names of all features for debugging/visualization"""
        names = []

        for period in self.sma_periods:
            names.append(f"sma_{period}_deviation")
        for period in self.ema_periods:
            names.append(f"ema_{period}_deviation")
        names.append("adx")
        names.append("ichimoku_conversion_dev")
        names.append("ichimoku_base_dev")

        names.append("rsi")
        names.append("macd")
        names.append("macd_histogram")
        names.append("stochastic_k")
        names.append("stochastic_d")
        names.append("roc_10")

        names.append("atr")
        names.append("bb_percent")
        names.append("bb_position")

        for period in self.volume_ma_periods:
            names.append(f"volume_ma_{period}_deviation")
        names.append("obv_zscore")
        names.append("vwap_deviation")

        names.extend(
            [
                "doji",
                "hammer",
                "inverted_hammer",
                "bullish_engulfing",
                "bearish_engulfing",
                "three_white_soldiers",
                "three_black_crows",
            ]
        )

        return names


def create_technical_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Create technical indicator features from OHLCV DataFrame.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        feature_cols: Specific columns to use (auto-detect if None)

    Returns:
        Array of technical features (shape: n_samples x n_features)
    """
    if feature_cols is None:
        required_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [col for col in required_cols if col in df.columns]
        if len(feature_cols) < 3:
            raise ValueError(
                f"DataFrame must contain at least open, high, low, close columns. Found: {feature_cols}"
            )

    generator = TechnicalIndicatorGenerator()

    open_ = df["open"] if "open" in df.columns else df["close"]
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    features = generator.compute_all_features(open_, high, low, close, volume)

    return features.combined_features


if __name__ == "__main__":
    np.random.seed(42)
    n_bars = 1000

    base_price = 17500
    returns = np.random.randn(n_bars) * 0.001
    close = pd.Series(base_price * (1 + returns).cumprod())
    high = close + np.random.rand(n_bars) * 10
    low = close - np.random.rand(n_bars) * 10
    open_ = close.shift(1).fillna(base_price)
    volume = pd.Series(np.random.randint(10000, 50000, n_bars))

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    generator = TechnicalIndicatorGenerator()
    features = generator.compute_all_features(
        df["open"], df["high"], df["low"], df["close"], df["volume"]
    )

    print(f"Trend features shape: {features.trend_features.shape}")
    print(f"Momentum features shape: {features.momentum_features.shape}")
    print(f"Volatility features shape: {features.volatility_features.shape}")
    print(f"Volume features shape: {features.volume_features.shape}")
    print(f"Pattern features shape: {features.pattern_features.shape}")
    print(f"Combined features shape: {features.combined_features.shape}")
    print(f"\nFeature names: {generator.get_feature_names()}")
