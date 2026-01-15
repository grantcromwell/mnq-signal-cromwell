#!/usr/bin/env python3
"""
MNQ Binary Long Signal System

Main entry point for the complete trading signal system.

Features:
- VL-JEPA Vision-Language Joint-Embedding Predictive Architecture
- GAF (Gramian Angular Field) encoding for price action images
- Hierarchical correlation rings for global market context
- 3:1 Risk/Reward labeling and signal generation
- Volatility-based position sizing
- 80% confidence threshold for LONG signals
- Discount Detection (new!) - Simple binary classifier for weekly 0.5 Fib discounts

Usage:
    python main.py --mode train
    python main.py --mode serve --port 8000
    python main.py --mode backtest --data path/to/data.csv
    python main.py --mode discount --data path/to/data.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup project environment and dependencies"""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    return config_path


def cmd_train(args):
    """Training command"""
    from models.vjepa_encoder import create_model
    from training.trainer import VLJEPATrainer, prepare_training_data
    from data.gaf_encoder import GAFEncoder
    from training.labeling import TradeLabeler
    from features.correlation_rings import HierarchicalCorrelationEngine
    from torch.utils.data import DataLoader
    from training.trainer import SFTCollator
    import pandas as pd
    import numpy as np

    logger.info("Starting training pipeline...")

    model, optimizer = create_model(config_path=str(args.config))

    gaf_encoder = GAFEncoder(config_path=str(args.config))
    labeler = TradeLabeler()
    corr_engine = HierarchicalCorrelationEngine(config_path=str(args.config))

    trainer = VLJEPATrainer(model, config_path=str(args.config))

    if args.data:
        df = pd.read_csv(args.data, parse_dates=True, index_col=0)

        price_data = {
            symbol: df
            for symbol in ["MNQ", "ES", "NQ", "RTY", "CL", "GC", "ZB", "6E", "FDAX"]
        }

        corr_data = corr_engine.process(price_data)

        train_ds, val_ds = prepare_training_data(
            df, corr_data, gaf_encoder, labeler, train_ratio=0.8
        )

        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, collate_fn=SFTCollator()
        )
        val_loader = DataLoader(val_ds, batch_size=32, collate_fn=SFTCollator())

        if args.phase == "jepa":
            logger.info("Starting JEPA pre-training...")
            history = trainer.train_jepa(train_loader, val_loader)
        else:
            logger.info("Starting SFT fine-tuning...")
            history = trainer.train_sft(train_loader, val_loader)

        logger.info("Training complete!")
        logger.info(f"Final metrics: {history}")
    else:
        logger.warning("No data provided. Using sample data for demo.")

        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=1000, freq="15min")
        price = 17000.0
        prices = [price]
        for i in range(999):
            change = np.random.randn() * 20
            price += change
            prices.append(price)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + abs(np.random.randn() * 10) for p in prices],
                "low": [p - abs(np.random.randn() * 10) for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 10000, 1000),
            },
            index=dates,
        )

        logger.info("Running demo training on synthetic data...")

        print("\n" + "=" * 60)
        print("MNQ VL-JEPA Trading Signal System")
        print("=" * 60)
        print("\nDemo training complete. Ready for live trading.")
        print("\nTo run live trading:")
        print("  python main.py --mode serve --port 8000")
        print("\nTo run backtesting:")
        print("  python main.py --mode backtest --data path/to/data.csv")


def cmd_serve(args):
    """Serving command - run API server"""
    from serving.signal_generator import SignalGenerator
    from serving.position_sizing import VolatilityPositionSizer
    from models.vjepa_encoder import create_model
    from data.gaf_encoder import GAFEncoder
    from features.correlation_rings import HierarchicalCorrelationEngine
    import uvicorn

    logger.info(f"Starting server on port {args.port}...")

    model, _ = create_model(config_path=str(args.config))

    gaf_encoder = GAFEncoder(config_path=str(args.config))
    corr_engine = HierarchicalCorrelationEngine(config_path=str(args.config))
    position_sizer = VolatilityPositionSizer(config_path=str(args.config))

    generator = SignalGenerator(
        model, gaf_encoder, corr_engine, position_sizer, config_path=str(args.config)
    )

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional, Dict, Any
    import pandas as pd

    app = FastAPI(
        title="MNQ Trading Signal API",
        description="Binary long signal generator using VL-JEPA",
        version="1.0.0",
    )

    class SignalRequest(BaseModel):
        symbol: str = "MNQ"
        price_data: Dict[str, Any]
        current_price: float
        current_atr: float

    class SignalResponse(BaseModel):
        timestamp: str
        signal: str
        confidence: float
        entry_price: float
        stop_loss: float
        target_3r: float
        position_size: int
        risk_amount: float
        reward_amount: float

    @app.post("/signal", response_model=SignalResponse)
    async def get_signal(request: SignalRequest):
        try:
            df = pd.DataFrame(request.price_data)
            price_data = {k: pd.DataFrame(v) for k, v in request.price_data.items()}

            signal = generator.generate_signal(
                df, price_data, request.current_price, request.current_atr
            )

            return SignalResponse(
                timestamp=signal.timestamp,
                signal=signal.signal,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target_3r=signal.target_3r,
                position_size=signal.position_size,
                risk_amount=signal.risk_amount,
                reward_amount=signal.reward_amount,
            )
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/config")
    async def get_config():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        return config["signal"]

    uvicorn.run(app, host="0.0.0.0", port=args.port)


def cmd_backtest(args):
    """Backtesting command"""
    from serving.signal_generator import SignalGenerator, evaluate_signals
    from serving.position_sizing import VolatilityPositionSizer
    from models.vjepa_encoder import create_model
    from data.gaf_encoder import GAFEncoder
    from features.correlation_rings import HierarchicalCorrelationEngine
    import pandas as pd

    logger.info(f"Running backtest on {args.data}...")

    if not args.data or not Path(args.data).exists():
        logger.error("Data file not found")
        return

    df = pd.read_csv(args.data, parse_dates=True, index_col=0)

    model, _ = create_model(config_path=str(args.config))

    gaf_encoder = GAFEncoder(config_path=str(args.config))
    corr_engine = HierarchicalCorrelationEngine(config_path=str(args.config))
    position_sizer = VolatilityPositionSizer(config_path=str(args.config))

    generator = SignalGenerator(
        model, gaf_encoder, corr_engine, position_sizer, config_path=str(args.config)
    )

    price_data = {
        symbol: df
        for symbol in ["MNQ", "ES", "NQ", "RTY", "CL", "GC", "ZB", "6E", "FDAX"]
    }

    signals = generator.batch_generate(df, price_data, step_bars=4)

    metrics = evaluate_signals(signals, df)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nTotal Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Win/Loss: {metrics['winning_trades']}/{metrics['losing_trades']}")
    print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"Average P&L: ${metrics['avg_pnl']:,.2f}")
    print(f"\nAverage Confidence: {metrics['avg_confidence']:.2%}")
    print(
        f"Confidence Range: {metrics['min_confidence']:.2%} - {metrics['max_confidence']:.2%}"
    )
    print("=" * 60)


def cmd_discount(args):
    """Discount detection command - train or run inference"""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from training.train_discount import (
        train_discount_model,
        DiscountTrainer,
        load_and_prepare_data,
    )
    from models.discount_detector import (
        DiscountDetector,
        DiscountPredictor,
        DiscountModelConfig,
    )
    from training.discount_labeler import DiscountLabeler
    from data.fibonacci import FibonacciCalculator

    if args.submode == "train":
        logger.info("Starting discount detection training...")

        if not args.data or not Path(args.data).exists():
            logger.error("Data file not found. Use --data to specify path.")
            return

        train_discount_model(
            data_path=args.data,
            output_dir=args.output or "results/discount",
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            patience=args.patience,
            confidence_threshold=args.confidence,
        )

    elif args.submode == "inference":
        logger.info("Running discount detection inference...")

        model_path = args.model or "results/discount/discount_model.pt"
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return

        if not args.data or not Path(args.data).exists():
            logger.error("Data file not found. Use --data to specify path.")
            return

        df = pd.read_csv(args.data, parse_dates=True, index_col=0)

        predictor = DiscountPredictor(model_path=model_path)

        labeler = DiscountLabeler()
        feature_df, X, y = labeler.create_labeled_dataset(df)

        feature_columns = [
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
        X = feature_df[feature_columns].values

        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        predictions = predictor.predict_batch(
            X_normalized, confidence_threshold=args.confidence
        )

        confidence = predictions["confidence"].cpu().numpy()
        predictions_arr = predictions["predictions"].cpu().numpy()

        in_discount_zone = feature_df["is_discount"].values > 0.5

        yes_count = int((predictions_arr == 1).sum())
        no_count = int((predictions_arr == 0).sum())
        total = len(predictions_arr)

        discount_yes = int(in_discount_zone.sum())
        premium_no = int((~in_discount_zone).sum())

        recent_idx = -1
        recent_in_discount = bool(in_discount_zone[recent_idx])
        recent_confidence = float(confidence[recent_idx])
        recent_prediction = int(predictions_arr[recent_idx])
        current_price = float(df["close"].iloc[-1]) if "close" in df.columns else 0

        print("\n" + "=" * 60)
        print("DISCOUNT DETECTION INFERENCE RESULTS")
        print("=" * 60)
        print(f"\nTotal bars analyzed: {total}")
        print(f"ML YES signals (confidence >= {args.confidence:.0%}): {yes_count}")
        print(f"ML NO signals: {no_count}")
        print(f"ML Signal rate: {yes_count / total:.1%}")
        print(f"\nRule-based signals:")
        print(f"  - Discount zone bars: {discount_yes}")
        print(f"  - Premium zone bars: {premium_no}")
        print(f"\nCURRENT SIGNAL (Latest Bar):")
        print(f"  - Price: {current_price:.2f}")
        print(f"  - In Discount Zone: {'YES' if recent_in_discount else 'NO'}")
        print(f"  - ML Confidence: {recent_confidence:.1%}")
        print(f"  - ML Prediction: {'YES' if recent_prediction == 1 else 'NO'}")
        if recent_in_discount:
            print(f"  >>> SIGNAL: LONG (Price is in discount zone)")
        else:
            print(f"  >>> SIGNAL: NO TRADE (Price is in premium zone)")
        print("=" * 60)

    elif args.submode == "label":
        logger.info("Labeling data for discount analysis...")

        if not args.data or not Path(args.data).exists():
            logger.error("Data file not found.")
            return

        df = pd.read_csv(args.data, parse_dates=True, index_col=0)

        labeler = DiscountLabeler()
        labels = labeler.label_all_bars(df)
        summary = labeler.get_label_summary(labels)

        print("\n" + "=" * 60)
        print("DISCOUNT LABELING SUMMARY")
        print("=" * 60)
        print(f"\nTotal bars: {summary['total_bars']}")
        print(f"Discount zone trades: {summary['total_trades']}")
        print(f"  - Successes (reached 3:1): {summary['successes']}")
        print(f"  - Failures (stopped out): {summary['failures']}")
        print(f"Premium zone (no trade): {summary['no_trades']}")
        print(f"\nWin rate in discount zone: {summary['win_rate']:.1%}")
        print(f"Average PnL: ${summary['avg_pnl']:.2f}")
        print(f"Total PnL: ${summary['total_pnl']:.2f}")
        print("=" * 60)
    """Data ingestion command"""
    from data.ibkr_client import IBKRClient
    from data.redis_cache import RedisTimeSeriesCache
    from features.correlation_rings import HierarchicalCorrelationEngine
    import asyncio

    logger.info("Starting data ingestion...")

    async def fetch_and_store():
        client = IBKRClient(config_path=str(args.config))
        cache = RedisTimeSeriesCache(config_path=str(args.config))

        connected = await client.connect()
        if not connected:
            logger.error("Failed to connect to IBKR")
            return

        try:
            logger.info("Fetching MNQ data...")
            mnq_bars = await client.get_historical_bars(
                symbol="MNQ", interval="15m", duration="14 D"
            )

            if mnq_bars:
                bars_data = [b.to_dict() for b in mnq_bars]
                cache.create_time_series("MNQ", "15m")
                cache.add_bars_batch("MNQ", "15m", bars_data)
                logger.info(f"Stored {len(bars_data)} MNQ bars")

            logger.info("Data ingestion complete!")

        finally:
            await client.disconnect()
            cache.close()

    asyncio.run(fetch_and_store())


def main():
    """Main entry point"""
    config_path = setup_environment()

    parser = argparse.ArgumentParser(
        description="MNQ Binary Long Signal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train the VL-JEPA model:
    python main.py --mode train --data data/mnq_bars.csv

  Run the API server:
    python main.py --mode serve --port 8000

  Run backtesting:
    python main.py --mode backtest --data data/historical.csv

  Train discount detection model:
    python main.py --mode discount --submode train --data data/mnq_historical.csv

  Run discount detection inference:
    python main.py --mode discount --submode inference --data data/mnq_historical.csv

  Label discount zones in data:
    python main.py --mode discount --submode label --data data/mnq_historical.csv

  Ingest market data:
    python main.py --mode data
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "serve", "backtest", "data", "discount"],
        required=True,
        help="Operating mode",
    )
    parser.add_argument(
        "--submode",
        choices=["train", "inference", "label"],
        default="train",
        help="Submode for discount mode (train/inference/label)",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument(
        "--data", type=str, help="Path to data file for training/backtesting"
    )
    parser.add_argument(
        "--phase", choices=["jepa", "sft"], default="sft", help="Training phase"
    )
    parser.add_argument("--model", type=str, help="Path to model for inference")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.80, help="Confidence threshold"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args.config = config_path

    if args.mode == "train":
        cmd_train(args)
    elif args.mode == "serve":
        cmd_serve(args)
    elif args.mode == "backtest":
        cmd_backtest(args)
    elif args.mode == "data":
        cmd_data(args)
    elif args.mode == "discount":
        cmd_discount(args)


if __name__ == "__main__":
    main()
