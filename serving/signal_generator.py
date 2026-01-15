"""
Signal Generator for MNQ Trading System

Generates real-time trading signals based on VL-JEPA model output.
Implements confidence-based filtering:
- LONG only when confidence >= 80%
- MONITOR when 50% <= confidence < 80%
- FLAT when confidence < 50%

Outputs signal with confidence score and position size.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal output"""

    timestamp: str
    symbol: str
    signal: str  # LONG, MONITOR, FLAT
    confidence: float
    entry_price: float
    stop_loss: float
    target_3r: float
    position_size: int
    risk_amount: float
    reward_amount: float
    model_version: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_3r": self.target_3r,
            "position_size": self.position_size,
            "risk_amount": self.risk_amount,
            "reward_amount": self.reward_amount,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SignalGenerator:
    """
    Real-time signal generator using trained VL-JEPA model.

    Process flow:
    1. Get latest price data from cache
    2. Generate GAF images
    3. Compute correlation rings
    4. Run model inference
    5. Apply confidence threshold (>=80% for LONG)
    6. Calculate position size
    7. Output trading signal
    """

    def __init__(
        self,
        model: torch.nn.Module,
        gaf_encoder: Any,
        correlation_engine: Any,
        position_sizer: Any,
        config_path: str = "../config/config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.gaf_encoder = gaf_encoder
        self.correlation_engine = correlation_engine
        self.position_sizer = position_sizer
        self.device = device

        with open(config_path, "r") as f:
            self.config = yaml.safe_load()["signal"]

        self.confidence_threshold_long = self.config["confidence_threshold_long"]
        self.confidence_threshold_low = self.config["confidence_threshold_low"]
        self.symbol = "MNQ"

        logger.info(
            f"SignalGenerator initialized with LONG threshold: {self.confidence_threshold_long}"
        )

    @torch.no_grad()
    def generate_signal(
        self,
        mnq_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        current_price: float,
        current_atr: float,
        model_version: str = "1.0.0",
    ) -> TradingSignal:
        """
        Generate trading signal for current market conditions.

        Args:
            mnq_df: Recent MNQ OHLCV DataFrame
            price_data: Dict of all asset price DataFrames
            current_price: Current MNQ price
            current_atr: Current ATR value
            model_version: Model version string

        Returns:
            TradingSignal object
        """
        timestamp = datetime.now().isoformat()

        images_3day = self._prepare_gaf_image(mnq_df.tail(78))
        images_week = self._prepare_week_gaf(mnq_df)
        text_embedding = self._prepare_correlation_text(price_data)

        images_3day = images_3day.to(self.device)
        images_week = images_week.to(self.device)
        text_embedding = text_embedding.to(self.device)

        output = self.model.forward_inference(
            images_3day,
            images_week,
            text_embedding,
            text_embedding,
            confidence_threshold=self.confidence_threshold_long,
        )

        signal_decision = output["signal"]
        confidence = output["confidence"]

        if signal_decision == "LONG" and current_atr > 0:
            stop_loss, target_3r = self._calculate_levels(
                current_price, current_atr, "long"
            )
            position_size, risk_amount, reward_amount = self.position_sizer.calculate(
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                atr=current_atr,
            )
        else:
            stop_loss = current_price - current_atr
            target_3r = current_price + (current_atr * 3)
            position_size = 0
            risk_amount = 0.0
            reward_amount = 0.0

        signal = TradingSignal(
            timestamp=timestamp,
            symbol=self.symbol,
            signal=signal_decision,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_3r=target_3r,
            position_size=position_size,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            model_version=model_version,
            metadata={
                "logits": output["logits"],
                "probabilities": output["probabilities"],
                "pred_class": output["pred_class"],
            },
        )

        logger.info(
            f"Signal generated: {signal.signal} | "
            f"Confidence: {signal.confidence:.2%} | "
            f"Position: {signal.position_size} contracts | "
            f"Entry: {signal.entry_price:.0f} | "
            f"SL: {signal.stop_loss:.0f} | "
            f"T3R: {signal.target_3r:.0f}"
        )

        return signal

    def _prepare_gaf_image(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare 3-day GAF image tensor"""
        gaf = self.gaf_encoder.create_3day_gaf(df)
        tensor = torch.from_numpy(gaf).float().unsqueeze(0)
        return tensor

    def _prepare_week_gaf(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare week GAF image tensor"""
        from data.gaf_encoder import resample_to_daily_bars

        daily_df = resample_to_daily_bars(df)
        gaf = self.gaf_encoder.create_week_gaf(daily_df)
        tensor = torch.from_numpy(gaf).float().unsqueeze(0)
        return tensor

    def _prepare_correlation_text(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> torch.Tensor:
        """Prepare correlation text embedding tensor"""
        corr_data = self.correlation_engine.process(price_data)

        from features.correlation_rings import flatten_correlation_for_bert

        embedding = flatten_correlation_for_bert(corr_data)

        tensor = torch.from_numpy(embedding).float().unsqueeze(0)
        return tensor

    def _calculate_levels(
        self, entry_price: float, atr: float, direction: str = "long"
    ) -> Tuple[float, float]:
        """Calculate stop loss and 3R target levels"""
        sl_multiplier = 1.0

        if direction == "long":
            stop_loss = entry_price - (atr * sl_multiplier)
            target_3r = entry_price + (atr * 3)
        else:
            stop_loss = entry_price + (atr * sl_multiplier)
            target_3r = entry_price - (atr * 3)

        return stop_loss, target_3r

    def batch_generate(
        self, df: pd.DataFrame, price_data: Dict[str, pd.DataFrame], step_bars: int = 4
    ) -> List[TradingSignal]:
        """
        Generate signals for historical bars (for backtesting).

        Args:
            df: Full OHLCV DataFrame
            price_data: Dict of all asset price DataFrames
            step_bars: Bars between signals

        Returns:
            List of TradingSignal objects
        """
        signals = []
        atr_calc = ATRCalculator(14)

        bars_per_day = 26
        window_bars = 3 * bars_per_day

        for i in range(window_bars, len(df) - 1, step_bars):
            window_df = df.iloc[i - window_bars : i]
            current_price = df["close"].iloc[i]
            current_atr = atr_calc.compute(df).iloc[i]

            signal = self.generate_signal(
                window_df, price_data, current_price, current_atr
            )
            signals.append(signal)

        logger.info(f"Generated {len(signals)} signals for backtest")
        return signals


class ATRCalculator:
    """Simple ATR calculator for signal generation"""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=self.period).mean()

        return atr


def evaluate_signals(signals: List[TradingSignal], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate signal performance on historical data.

    Args:
        signals: List of TradingSignal objects
        df: Full OHLCV DataFrame with index matching signal timestamps

    Returns:
        Dict of performance metrics
    """
    if not signals:
        return {"error": "No signals to evaluate"}

    trades = []
    for signal in signals:
        if signal.signal != "LONG" or signal.position_size == 0:
            continue

        entry_idx = df.index.get_indexer(
            [pd.to_datetime(signal.timestamp)], method="nearest"
        )[0]
        if entry_idx >= len(df) - 1:
            continue

        future_bars = df.iloc[entry_idx + 1 :]

        hit_stop = False
        hit_target = False
        exit_price = future_bars["close"].iloc[-1]

        for _, row in future_bars.iterrows():
            if row["low"] <= signal.stop_loss:
                hit_stop = True
                exit_price = signal.stop_loss
                break
            if row["high"] >= signal.target_3r:
                hit_target = True
                exit_price = signal.target_3r
                break

        pnl_per_contract = (exit_price - signal.entry_price) * 0.5

        trades.append(
            {
                "timestamp": signal.timestamp,
                "entry": signal.entry_price,
                "exit": exit_price,
                "stop_loss": signal.stop_loss,
                "target_3r": signal.target_3r,
                "hit_stop": hit_stop,
                "hit_target": hit_target,
                "confidence": signal.confidence,
                "position_size": signal.position_size,
                "pnl": pnl_per_contract * signal.position_size,
            }
        )

    if not trades:
        return {"error": "No completed trades"}

    trades_df = pd.DataFrame(trades)

    metrics = {
        "total_trades": len(trades),
        "winning_trades": int((trades_df["pnl"] > 0).sum()),
        "losing_trades": int((trades_df["pnl"] <= 0).sum()),
        "win_rate": float((trades_df["pnl"] > 0).mean()),
        "avg_pnl": float(trades_df["pnl"].mean()),
        "total_pnl": float(trades_df["pnl"].sum()),
        "avg_confidence": float(trades_df["confidence"].mean()),
        "max_confidence": float(trades_df["confidence"].max()),
        "min_confidence": float(trades_df["confidence"].min()),
    }

    winning_trades_pnl = trades_df[trades_df["pnl"] > 0]["pnl"]
    losing_trades_pnl = trades_df[trades_df["pnl"] <= 0]["pnl"]

    metrics["avg_win"] = (
        float(winning_trades_pnl.mean()) if len(winning_trades_pnl) > 0 else 0
    )
    metrics["avg_loss"] = (
        float(losing_trades_pnl.mean()) if len(losing_trades_pnl) > 0 else 0
    )

    if metrics["avg_loss"] != 0:
        metrics["profit_factor"] = (
            abs(winning_trades_pnl.sum() / losing_trades_pnl.sum())
            if losing_trades_pnl.sum() != 0
            else float("inf")
        )

    return metrics


if __name__ == "__main__":
    from models.vjepa_encoder import create_model
    from data.gaf_encoder import GAFEncoder
    from features.correlation_rings import HierarchicalCorrelationEngine
    from serving.position_sizing import VolatilityPositionSizer

    model, _ = create_model()

    gaf_encoder = GAFEncoder()
    corr_engine = HierarchicalCorrelationEngine()
    position_sizer = VolatilityPositionSizer()

    generator = SignalGenerator(model, gaf_encoder, corr_engine, position_sizer)

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
            "open": prices,
            "high": [p + abs(np.random.randn() * 10) for p in prices],
            "low": [p - abs(np.random.randn() * 10) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, 500),
        },
        index=dates,
    )

    price_data = {symbol: df for symbol in ["MNQ", "ES", "NQ", "RTY", "CL", "GC"]}

    current_price = df["close"].iloc[-1]
    current_atr = 50.0

    signal = generator.generate_signal(
        df.tail(100), price_data, current_price, current_atr
    )

    print(f"\nTrading Signal:")
    print(f"  Signal: {signal.signal}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Entry: {signal.entry_price:.0f}")
    print(f"  Stop Loss: {signal.stop_loss:.0f}")
    print(f"  Target 3R: {signal.target_3r:.0f}")
    print(f"  Position Size: {signal.position_size} contracts")
    print(f"  Risk Amount: ${signal.risk_amount:.2f}")
    print(f"  Reward Amount: ${signal.reward_amount:.2f}")
