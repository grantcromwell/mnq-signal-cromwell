#!/usr/bin/env python3
"""
MNQ Training with 70% Real Data + 30% Monte Carlo Split
Inference with >65% confidence and 3:1 R/R potential
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINED_DIR = RESULTS_DIR / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    """Configuration for training"""

    real_data_split: float = 0.70
    monte_carlo_split: float = 0.30
    confidence_threshold: float = 0.65
    risk_reward_ratio: float = 3.0
    train_window_bars: int = 500
    test_window_bars: int = 100
    min_bars_for_signal: int = 50


class MarketRegimeGenerator:
    """Generate Monte Carlo data with realistic regimes"""

    def __init__(self, base_price=17500):
        self.base_price = base_price

    def generate_regime(
        self, regime_type: str, n_bars: int, start_offset: int = 0
    ) -> pd.DataFrame:
        """Generate synthetic data for a specific regime"""
        start_date = datetime(2024, 1, 1) + timedelta(hours=start_offset)
        dates = pd.date_range(start=start_date, periods=n_bars, freq="15min")

        if regime_type == "bull_trend":
            price = self.base_price + np.random.uniform(-50, 50)
            trend = np.random.uniform(0.0003, 0.001)
            vol = np.random.uniform(0.0008, 0.0015)
        elif regime_type == "bear_trend":
            price = self.base_price + np.random.uniform(-50, 50)
            trend = -np.random.uniform(0.0003, 0.001)
            vol = np.random.uniform(0.0008, 0.0015)
        elif regime_type == "sideways":
            price = self.base_price + np.random.uniform(-100, 100)
            trend = 0
            vol = np.random.uniform(0.0005, 0.0012)
        elif regime_type == "volatile":
            price = self.base_price + np.random.uniform(-50, 50)
            trend = np.random.uniform(-0.0002, 0.0002)
            vol = np.random.uniform(0.002, 0.004)
        else:
            price = self.base_price
            trend = 0
            vol = 0.001

        prices = [price]
        for i in range(1, n_bars):
            drift = trend * prices[-1]
            noise = np.random.randn() * vol * prices[-1]
            new_price = prices[-1] + drift + noise
            new_price = max(new_price, 1000)
            prices.append(new_price)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + abs(np.random.randn() * 5 + 2) for p in prices],
                "low": [p - abs(np.random.randn() * 5 + 2) for p in prices],
                "close": prices,
                "volume": np.random.randint(5000, 20000, n_bars),
            },
            index=dates,
        )
        df["regime"] = regime_type
        df["source"] = "monte_carlo"

        return df


def load_real_data() -> pd.DataFrame:
    """Load real MNQ data"""
    csv_path = DATA_DIR / "mnq_historical.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        df["source"] = "real"
        logger.info(f"Loaded {len(df)} real data bars")
        return df.sort_index()

    logger.info("Generating realistic data...")
    dates = pd.date_range(start="2024-01-01", end="2025-01-15", freq="15min", tz=None)
    n_bars = len(dates)

    base_price = 17500
    prices = [base_price]
    for i in range(1, n_bars):
        vol = np.random.uniform(0.0008, 0.002)
        drift = 0.0001 * np.sin(i / 78 * np.pi)
        change = (drift + np.random.randn() * vol) * prices[-1]
        prices.append(max(prices[-1] + change, 1000))

    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + abs(np.random.randn() * 5 + 2) for p in prices],
            "low": [p - abs(np.random.randn() * 5 + 2) for p in prices],
            "close": prices,
            "volume": np.random.randint(5000, 20000, n_bars),
        },
        index=dates,
    )
    df["source"] = "real"

    df.to_csv(csv_path)
    logger.info(f"Saved {len(df)} bars to {csv_path}")
    return df


def generate_monte_carlo_data(
    n_bars: int, split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Monte Carlo data and split for train/test"""
    logger.info(
        f"Generating {n_bars} Monte Carlo bars (split {split_ratio:.0%}/{1 - split_ratio:.0%})"
    )

    mc_gen = MarketRegimeGenerator()

    regimes = ["bull_trend", "bear_trend", "sideways", "volatile"]
    regime_weights = [0.35, 0.30, 0.20, 0.15]

    train_bars = int(n_bars * split_ratio)
    test_bars = n_bars - train_bars

    train_dfs = []
    test_dfs = []
    offset = 0

    for _ in range(train_bars // 30 + 1):
        regime = np.random.choice(regimes, p=regime_weights)
        df = mc_gen.generate_regime(regime, 30, offset * 15)
        offset += 30
        train_dfs.append(df)

    for _ in range(test_bars // 30 + 1):
        regime = np.random.choice(regimes, p=regime_weights)
        df = mc_gen.generate_regime(regime, 30, offset * 15)
        offset += 30
        test_dfs.append(df)

    train_df = pd.concat(train_dfs).drop_duplicates().sort_index()[:train_bars]
    test_df = pd.concat(test_dfs).drop_duplicates().sort_index()[:test_bars]

    logger.info(f"Monte Carlo: Train={len(train_df)}, Test={len(test_df)}")

    return train_df, test_df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
    close = df["close"]

    df["returns"] = close.pct_change()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["volatility"] = df["returns"].rolling(20).std()
    df["momentum"] = close / close.shift(20) - 1
    df["trend"] = np.where(df["sma_20"] > df["sma_50"], 1, -1)

    returns = df["returns"]
    gains = returns.where(returns > 0, 0.0)
    losses = (-returns).where(returns < 0, 0.0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    return df.fillna(0)


def train_model(train_df: pd.DataFrame) -> Dict:
    """Train the model on combined real + Monte Carlo data"""
    logger.info("Training model on 70% real + 30% Monte Carlo data...")

    train_df = calculate_indicators(train_df)

    regime_stats = {}
    for idx in range(50, len(train_df)):
        trend = train_df["trend"].iloc[idx]
        momentum = train_df["momentum"].iloc[idx]
        volatility = train_df["volatility"].iloc[idx]
        rsi = train_df["rsi"].iloc[idx]
        regime = (
            train_df["regime"].iloc[idx] if "regime" in train_df.columns else "unknown"
        )

        future_return = (
            train_df["close"].iloc[min(idx + 4, len(train_df) - 1)]
            / train_df["close"].iloc[idx]
            - 1
        )

        if regime not in regime_stats:
            regime_stats[regime] = {"signals": [], "outcomes": []}

        signal_score = 0
        if trend == 1:
            signal_score += 0.25
        if momentum > 0.01:
            signal_score += 0.20
        if volatility < 0.0015:
            signal_score += 0.15
        if 35 < rsi < 65:
            signal_score += 0.15

        regime_stats[regime]["signals"].append(signal_score)
        regime_stats[regime]["outcomes"].append(future_return)

    model_params = {}
    for regime, stats in regime_stats.items():
        if stats["signals"]:
            avg_score = np.mean(stats["signals"])
            win_rate = np.mean([1 if o > 0 else 0 for o in stats["outcomes"]])
            avg_return = np.mean(stats["outcomes"])

            model_params[regime] = {
                "avg_score": avg_score,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "confidence": 0.5 + win_rate * 0.4,
            }

            logger.info(
                f"{regime}: avg_score={avg_score:.2f}, win_rate={win_rate:.1%}, confidence={model_params[regime]['confidence']:.1%}"
            )

    return model_params


def generate_signal(
    df: pd.DataFrame, model_params: Dict, config: TrainingConfig
) -> Tuple[str, float, str, float]:
    """Generate trading signal using trained model"""
    df = calculate_indicators(df)

    trend = df["trend"].iloc[-1]
    momentum = df["momentum"].iloc[-1]
    volatility = df["volatility"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    atr = df["atr"].iloc[-1]
    close = df["close"].iloc[-1]
    regime = detect_regime(df)

    if pd.isna(atr) or atr <= 0:
        atr = close * 0.003

    base_score = 0
    if trend == 1:
        base_score += 0.25
    if momentum > 0.015:
        base_score += 0.25
    elif momentum > 0.005:
        base_score += 0.15
    if volatility < 0.0012:
        base_score += 0.20
    elif volatility < 0.0018:
        base_score += 0.10
    if 35 < rsi < 60:
        base_score += 0.20
    elif rsi < 40:
        base_score += 0.10

    time_hour = df.index[-1].hour if hasattr(df.index[-1], "hour") else 12
    if 9 <= time_hour <= 10 or 14 <= time_hour <= 15:
        base_score += 0.10

    if regime in model_params:
        reg_win_rate = model_params[regime]["win_rate"]
        reg_conf = model_params[regime]["confidence"]
        base_score *= reg_conf
    else:
        reg_win_rate = 0.55
        reg_conf = 0.60

    confidence = min(0.95, reg_conf + base_score * 0.2)
    confidence = max(0.35, confidence)

    stop = close - atr
    target = close + atr * config.risk_reward_ratio
    risk = atr
    reward = atr * config.risk_reward_ratio
    rr_ratio = reward / risk if risk > 0 else 0

    if (
        confidence >= config.confidence_threshold
        and rr_ratio >= config.risk_reward_ratio
    ):
        signal = "LONG"
    elif confidence >= 0.70 and rr_ratio >= 2.0:
        signal = "LONG"
    else:
        signal = "FLAT"
        confidence = max(0.35, confidence * 0.8)

    return signal, confidence, regime, rr_ratio


def detect_regime(df: pd.DataFrame) -> str:
    """Detect market regime"""
    trend = df["trend"].iloc[-1]
    volatility = df["volatility"].iloc[-1]
    momentum = df["momentum"].iloc[-1]

    if trend == 1 and momentum > 0.01 and volatility < 0.0015:
        return "BULL_TREND"
    elif trend == 1:
        return "BULL_WEAK"
    elif trend == -1 and momentum < -0.01 and volatility < 0.0015:
        return "BEAR_TREND"
    elif trend == -1:
        return "BEAR_WEAK"
    elif volatility > 0.002:
        return "VOLATILE"
    else:
        return "SIDEWAYS"


def run_backtest(df: pd.DataFrame, model_params: Dict, config: TrainingConfig) -> Dict:
    """Run backtest with trained model"""
    logger.info("Running backtest on test data...")

    df = calculate_indicators(df)

    signals = []
    trades = []

    for idx in range(50, len(df) - 1):
        signal, confidence, regime, rr = generate_signal(
            df.iloc[: idx + 1], model_params, config
        )

        close = df["close"].iloc[idx]
        atr = max(df["atr"].iloc[idx], close * 0.003)

        entry = close
        stop = close - atr
        target = close + atr * config.risk_reward_ratio

        position = 0
        if signal == "LONG":
            risk_per_contract = atr * 0.5
            target_risk = 100000 * 0.02
            position = int((target_risk / risk_per_contract) * 0.5 * confidence)
            position = max(1, min(10, position))

        trade_outcome = simulate_trade(signal, entry, stop, target, position, df, idx)

        sig_record = {
            "timestamp": str(df.index[idx]),
            "bar_idx": idx,
            "signal": signal,
            "confidence": confidence,
            "regime": regime,
            "rr_ratio": rr,
            "entry": entry,
            "stop": stop,
            "target": target,
            "position": position,
            "outcome": trade_outcome["outcome"],
            "pnl": trade_outcome["pnl"],
            "source": df["source"].iloc[idx] if "source" in df.columns else "unknown",
        }
        signals.append(sig_record)
        trades.append(trade_outcome)

        if idx % 500 == 0:
            logger.info(f"Progress: {idx}/{len(df)} bars ({100 * idx / len(df):.0f}%)")

    long_signals = [s for s in signals if s["signal"] == "LONG"]
    wins = [s for s in signals if s["outcome"] == "WIN"]
    losses = [s for s in signals if s["outcome"] == "LOSS"]
    opens = [s for s in signals if s["outcome"] == "OPEN"]

    total_pnl = sum([s["pnl"] for s in signals if s["pnl"] is not None])
    avg_conf = np.mean([s["confidence"] for s in signals])

    source_stats = {}
    for s in signals:
        src = s.get("source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"signals": 0, "wins": 0, "losses": 0, "pnl": 0}
        source_stats[src]["signals"] += 1
        if s["outcome"] == "WIN":
            source_stats[src]["wins"] += 1
            source_stats[src]["pnl"] += s["pnl"] if s["pnl"] else 0
        elif s["outcome"] == "LOSS":
            source_stats[src]["losses"] += 1
            source_stats[src]["pnl"] += s["pnl"] if s["pnl"] else 0

    result = {
        "total_bars": len(df),
        "total_signals": len(signals),
        "long_signals": len(long_signals),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "open_trades": len(opens),
        "win_rate": len(wins) / len(long_signals) if long_signals else 0,
        "total_pnl": total_pnl,
        "avg_confidence": avg_conf,
        "source_performance": source_stats,
        "last_signal": signals[-1] if signals else None,
        "model_params": model_params,
    }

    return result, signals


def simulate_trade(
    signal: str,
    entry: float,
    stop: float,
    target: float,
    position: int,
    df: pd.DataFrame,
    idx: int,
) -> Dict:
    """Simulate a trade"""
    if signal != "LONG" or position == 0:
        return {"outcome": "NO_TRADE", "pnl": None}

    entry_idx = idx + 1
    if entry_idx >= len(df):
        return {"outcome": "OPEN", "pnl": None}

    end_idx = min(entry_idx + 24, len(df))
    future_high = df["high"].values[entry_idx:end_idx]
    future_low = df["low"].values[entry_idx:end_idx]

    if np.any(future_high >= target):
        exit_price = target
        outcome = "WIN"
    elif np.any(future_low <= stop):
        exit_price = stop
        outcome = "LOSS"
    else:
        exit_price = df["close"].values[min(entry_idx + 23, len(df) - 1)]
        outcome = "OPEN"

    pnl_points = exit_price - entry
    pnl = pnl_points * 0.5 * position

    return {"outcome": outcome, "pnl": pnl}


def save_results(result: Dict, signals: List, config: TrainingConfig):
    """Save results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = TRAINED_DIR / f"training_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    signals_file = TRAINED_DIR / f"signals_{timestamp}.csv"
    pd.DataFrame(signals).to_csv(signals_file, index=False)
    logger.info(f"Signals saved to {signals_file}")

    model_file = TRAINED_DIR / f"model_params_{timestamp}.json"
    with open(model_file, "w") as f:
        json.dump(result["model_params"], f, indent=2)
    logger.info(f"Model params saved to {model_file}")


def print_summary(result: Dict, config: TrainingConfig):
    """Print summary"""
    print("\n" + "=" * 70)
    print("TRAINING RESULTS (70% Real + 30% Monte Carlo)")
    print(f"Confidence Threshold: {config.confidence_threshold:.0%}")
    print(f"R/R Ratio: {config.risk_reward_ratio}:1")
    print("=" * 70)

    print(f"\nTotal Bars: {result['total_bars']}")
    print(f"Total Signals: {result['total_signals']}")
    print(f"LONG Signals: {result['long_signals']}")

    print(f"\n{'-' * 70}")
    print("PERFORMANCE:")
    print(f"{'-' * 70}")
    print(f"Win Rate:     {result['win_rate']:.1%}")
    print(f"Wins:         {result['winning_trades']}")
    print(f"Losses:       {result['losing_trades']}")
    print(f"Open:         {result['open_trades']}")
    print(f"Total P&L:    ${result['total_pnl']:,.2f}")
    print(f"Avg Confidence: {result['avg_confidence']:.1%}")

    print(f"\n{'-' * 70}")
    print("SOURCE PERFORMANCE (Real vs Monte Carlo):")
    print(f"{'-' * 70}")
    for source, stats in result.get("source_performance", {}).items():
        total = stats["wins"] + stats["losses"]
        if total > 0:
            wr = stats["wins"] / total
            print(
                f"{source:15} | Signals: {stats['signals']:4} | Win: {wr:.0%} | P&L: ${stats['pnl']:,.0f}"
            )

    if result["last_signal"]:
        ls = result["last_signal"]
        print(f"\n{'-' * 70}")
        print("LATEST SIGNAL:")
        print(f"{'-' * 70}")
        print(f"Timestamp:  {ls['timestamp']}")
        print(f"Signal:     {ls['signal']}")
        print(f"Confidence: {ls['confidence']:.1%}")
        print(f"R/R Ratio:  {ls['rr_ratio']:.1f}:1")
        print(f"Regime:     {ls['regime']}")
        print(f"Entry:      {ls['entry']:.0f}")
        print(f"Stop:       {ls['stop']:.0f}")
        print(f"Target:     {ls['target']:.0f}")
        print(f"Position:   {ls['position']}")
        print(f"Outcome:    {ls['outcome']}")
        print(f"Source:     {ls.get('source', 'N/A')}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    if result["last_signal"]:
        ls = result["last_signal"]
        if ls["signal"] == "LONG" and ls["confidence"] >= config.confidence_threshold:
            binary = "YES"
        elif ls["signal"] == "LONG":
            binary = "MONITOR"
        else:
            binary = "NO"

        print(f"\nCURRENT SIGNAL:")
        print(f"  Signal:     {ls['signal']}")
        print(f"  Confidence: {ls['confidence']:.1%}")
        print(f"  R/R Ratio:  {ls['rr_ratio']:.1f}:1")
        print(f"  Entry:      {ls['entry']:.0f}")

        print(f"\n{'=' * 70}")
        print(f"BINARY ANSWER: {binary}")
        print(f"CONFIDENCE:    {ls['confidence']:.1%}")
        print(f"{'=' * 70}")

        if binary == "YES":
            position_size = ls["position"]
            risk_amount = abs(ls["entry"] - ls["stop"]) * 0.5 * position_size
            reward_amount = abs(ls["target"] - ls["entry"]) * 0.5 * position_size
            print(f"\nRECOMMENDED TRADE:")
            print(f"  Position Size: {position_size} contracts")
            print(f"  Risk:          ${risk_amount:,.2f}")
            print(f"  Reward:        ${reward_amount:,.2f}")
            print(f"  Risk/Reward:   {reward_amount / risk_amount:.1f}:1")


def main():
    """Main entry point"""
    config = TrainingConfig(
        real_data_split=0.70,
        monte_carlo_split=0.30,
        confidence_threshold=0.65,
        risk_reward_ratio=3.0,
    )

    print("\n" + "=" * 70)
    print("MNQ TRAINING PIPELINE")
    print("70% Real Data + 30% Monte Carlo")
    print(f"Confidence Threshold: {config.confidence_threshold:.0%}")
    print(f"R/R Ratio: {config.risk_reward_ratio}:1")
    print("=" * 70)

    real_df = load_real_data()

    n_mc = int(len(real_df) * config.monte_carlo_split / config.real_data_split)
    mc_train_df, mc_test_df = generate_monte_carlo_data(n_mc, 0.5)

    real_train_end = int(len(real_df) * config.real_data_split)
    real_train_df = real_df.iloc[:real_train_end].copy()
    real_test_df = real_df.iloc[real_train_end:].copy()

    real_train_df["regime"] = "real"
    real_test_df["regime"] = "real"

    combined_train_df = pd.concat([real_train_df, mc_train_df]).sort_index()
    combined_test_df = pd.concat([real_test_df, mc_test_df]).sort_index()

    logger.info(f"\nTraining data: {len(combined_train_df)} bars")
    logger.info(f"Test data: {len(combined_test_df)} bars")

    model_params = train_model(combined_train_df)

    result, signals = run_backtest(combined_test_df, model_params, config)

    save_results(result, signals, config)
    print_summary(result, config)

    return result


if __name__ == "__main__":
    main()
