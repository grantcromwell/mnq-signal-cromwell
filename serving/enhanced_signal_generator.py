"""
Enhanced Signal Generator with News Sentiment

Generates trading signals considering:
- Model confidence
- News sentiment (Alpha Vantage)
- Technical indicators
- Volume profile (from OHLCV data)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingTarget:
    """Trading target with source"""

    price: float
    strength: float
    source: str
    reason: str


@dataclass
class EnhancedSignal:
    """Enhanced trading signal with all context"""

    signal: str
    confidence: float
    binary_answer: str
    entry_price: float
    stop_loss: float
    target_3r: float
    primary_target: TradingTarget
    secondary_targets: List[TradingTarget]
    position_size: int
    atr: float
    sentiment_score: float
    regime: str
    reasons: List[str]
    timestamp: str
    risk_reward_actual: float


class EnhancedSignalGenerator:
    """
    Enhanced signal generator with news sentiment and technical analysis.
    """

    def __init__(
        self,
        confidence_threshold_long: float = 0.65,
        confidence_threshold_short: float = 0.70,
        sentiment_threshold: float = 0.1,
        risk_reward_ratio: float = 3.0,
        risk_per_trade: float = 0.20,
        max_position: int = 12,
        contract_multiplier: float = 0.5,
    ):
        self.confidence_threshold_long = confidence_threshold_long
        self.confidence_threshold_short = confidence_threshold_short
        self.sentiment_threshold = sentiment_threshold
        self.risk_reward_ratio = risk_reward_ratio
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.contract_multiplier = contract_multiplier

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
    ) -> int:
        """
        Calculate position size based on risk.
        """
        risk_amount = portfolio_value * self.risk_per_trade
        risk_per_contract = abs(entry_price - stop_loss) * self.contract_multiplier

        if risk_per_contract <= 0:
            return 1

        position = int(risk_amount / risk_per_contract)
        position = max(1, min(position, self.max_position))

        return position

    def find_targets_from_volume_profile(
        self,
        current_price: float,
        volume_profile: Dict,
        atr: float,
        direction: str = "long",
    ) -> List[TradingTarget]:
        """
        Find targets from volume profile analysis.
        """
        targets = []

        poc = volume_profile.get("poc", current_price)
        vah = volume_profile.get("value_area_high", current_price + atr * 2)
        val = volume_profile.get("value_area_low", current_price - atr * 2)

        if direction == "long":
            if poc > current_price:
                poc_strength = 0.8 * (1 - abs(poc - current_price) / (atr * 3))
                targets.append(
                    TradingTarget(
                        price=poc,
                        strength=max(0.3, poc_strength),
                        source="poc",
                        reason="Point of Control (institutional activity)",
                    )
                )

            if vah > current_price and vah > poc:
                vah_strength = 0.7 * (1 - abs(vah - current_price) / (atr * 4))
                targets.append(
                    TradingTarget(
                        price=vah,
                        strength=max(0.3, vah_strength),
                        source="value_area_high",
                        reason="Value Area High (resistance)",
                    )
                )
        else:
            if poc < current_price:
                poc_strength = 0.8 * (1 - abs(poc - current_price) / (atr * 3))
                targets.append(
                    TradingTarget(
                        price=poc,
                        strength=max(0.3, poc_strength),
                        source="poc",
                        reason="Point of Control (magnet effect)",
                    )
                )

            if val < current_price and val < poc:
                val_strength = 0.7 * (1 - abs(val - current_price) / (atr * 4))
                targets.append(
                    TradingTarget(
                        price=val,
                        strength=max(0.3, val_strength),
                        source="value_area_low",
                        reason="Value Area Low (support becomes resistance)",
                    )
                )

        targets.sort(key=lambda x: x.strength, reverse=True)
        return targets[:5]

    def generate_signal(
        self,
        current_price: float,
        atr: float,
        model_confidence: float,
        sentiment_score: float,
        volume_profile: Dict,
        technical_vec: np.ndarray,
        regime: str = "unknown",
        portfolio_value: float = 100000,
    ) -> EnhancedSignal:
        """
        Generate enhanced trading signal.
        """
        reasons = []
        signal = "FLAT"

        long_targets = self.find_targets_from_volume_profile(
            current_price, volume_profile, atr, direction="long"
        )
        short_targets = self.find_targets_from_volume_profile(
            current_price, volume_profile, atr, direction="short"
        )

        sentiment_contribution = sentiment_score * 0.3

        if sentiment_score >= self.sentiment_threshold:
            reasons.append(f"Bullish sentiment: {sentiment_score:.2f}")

        if (
            model_confidence >= self.confidence_threshold_long
            and model_confidence + sentiment_contribution > 0.5
        ):
            signal = "LONG"
            reasons.append(f"Model confidence: {model_confidence:.1%}")
            if long_targets:
                reasons.append(
                    f"Primary target: {long_targets[0].source} at {long_targets[0].price:.0f}"
                )

        elif (1 - model_confidence) >= self.confidence_threshold_short and (
            1 - model_confidence
        ) - sentiment_contribution > 0.5:
            signal = "SHORT"
            reasons.append(f"Model bearish: {(1 - model_confidence):.1%}")
            if short_targets:
                reasons.append(
                    f"Primary target: {short_targets[0].source} at {short_targets[0].price:.0f}"
                )

        elif model_confidence >= 0.55 and sentiment_score >= 0:
            signal = "LONG"
            reasons.append(f"Moderate confidence: {model_confidence:.1%}")

        if signal != "FLAT":
            if signal == "LONG":
                stop_loss = current_price - atr
                base_target = current_price + atr * self.risk_reward_ratio

                if long_targets:
                    best_target = min(base_target, long_targets[0].price)
                else:
                    best_target = base_target

                target_3r = best_target
                primary_target = (
                    long_targets[0]
                    if long_targets
                    else TradingTarget(
                        price=best_target,
                        strength=0.5,
                        source="model",
                        reason="Model-based target",
                    )
                )
                secondary_targets = long_targets[1:]

            else:
                stop_loss = current_price + atr
                base_target = current_price - atr * self.risk_reward_ratio

                if short_targets:
                    best_target = max(base_target, short_targets[0].price)
                else:
                    best_target = base_target

                target_3r = best_target
                primary_target = (
                    short_targets[0]
                    if short_targets
                    else TradingTarget(
                        price=best_target,
                        strength=0.5,
                        source="model",
                        reason="Model-based target",
                    )
                )
                secondary_targets = short_targets[1:]

            position_size = self.calculate_position_size(
                portfolio_value, current_price, stop_loss
            )
            risk_reward_actual = (
                abs(target_3r - current_price) / abs(current_price - stop_loss)
                if stop_loss != current_price
                else self.risk_reward_ratio
            )

        else:
            stop_loss = current_price
            target_3r = current_price
            position_size = 0
            primary_target = TradingTarget(
                price=current_price, strength=0, source="none", reason="No signal"
            )
            secondary_targets = []
            risk_reward_actual = 0

        binary_answer = "YES" if signal == "LONG" else "NO"

        return EnhancedSignal(
            signal=signal,
            confidence=model_confidence,
            binary_answer=binary_answer,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_3r=target_3r,
            primary_target=primary_target,
            secondary_targets=secondary_targets,
            position_size=position_size,
            atr=atr,
            sentiment_score=sentiment_score,
            regime=regime,
            reasons=reasons,
            timestamp=str(pd.Timestamp.utcnow()),
            risk_reward_actual=risk_reward_actual,
        )


def in_trading_hours(
    timestamp: pd.Timestamp, start_hour: int = 3, end_hour: int = 15
) -> bool:
    """
    Check if timestamp is within trading hours (EST).
    """
    try:
        est = timestamp.tz_localize("UTC").tz_convert("America/New_York")
        hour = est.hour
        return start_hour <= hour < end_hour
    except:
        hour = (
            timestamp.hour
            if hasattr(timestamp, "hour")
            else timestamp.to_pydatetime().hour
        )
        return start_hour <= hour < end_hour


def compute_volume_profile(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: pd.Series,
    price_bins: int = 50,
) -> Dict:
    """
    Compute volume profile from OHLCV data.
    """
    price_range = highs.max() - lows.min()
    bin_size = price_range / price_bins

    price_centers = np.linspace(
        lows.min() + bin_size / 2,
        highs.max() - bin_size / 2,
        price_bins,
    )

    volume_at_price = np.zeros(price_bins)

    for i in range(len(closes)):
        high_idx = min(int((highs.iloc[i] - lows.min()) / bin_size), price_bins - 1)
        low_idx = min(int((lows.iloc[i] - lows.min()) / bin_size), price_bins - 1)

        avg_volume = volumes.iloc[i] / max(high_idx - low_idx + 1, 1)
        volume_at_price[low_idx : high_idx + 1] += avg_volume

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
        "total_volume": volume_at_price.sum(),
    }


if __name__ == "__main__":
    generator = EnhancedSignalGenerator(
        risk_per_trade=0.20,
        max_position=12,
    )

    current_price = 17500
    atr = 61.4
    model_confidence = 0.78
    sentiment_score = 0.25
    portfolio_value = 100000

    volume_profile = {
        "poc": 17525,
        "value_area_high": 17580,
        "value_area_low": 17450,
        "volume_asymmetry": 1.8,
    }

    technical_vec = np.random.randn(30)

    signal = generator.generate_signal(
        current_price=current_price,
        atr=atr,
        model_confidence=model_confidence,
        sentiment_score=sentiment_score,
        volume_profile=volume_profile,
        technical_vec=technical_vec,
        regime="bull_weak",
        portfolio_value=portfolio_value,
    )

    print(f"\n=== ENHANCED SIGNAL ===")
    print(f"Signal: {signal.signal}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Binary: {signal.binary_answer}")
    print(f"Entry: {signal.entry_price:.0f}")
    print(f"Stop: {signal.stop_loss:.0f}")
    print(f"Target: {signal.target_3r:.0f}")
    print(f"Position: {signal.position_size} contracts")
    print(f"R/R: {signal.risk_reward_actual:.1f}:1")
    print(f"\nReasons:")
    for r in signal.reasons:
        print(f"  - {r}")
    print(
        f"\nPrimary Target: {signal.primary_target.source} at {signal.primary_target.price:.0f}"
    )
    print(f"  Strength: {signal.primary_target.strength:.2f}")
    print(f"  Reason: {signal.primary_target.reason}")
