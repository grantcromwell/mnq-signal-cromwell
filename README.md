# MNQ Discount Signal System - Version 1

Binary trading signal system for Micro Nasdaq 100 (MNQ) futures using weekly 0.5 Fibonacci discount detection.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch live data and get signal
python quick_signal.py

# Record trade (if signal is YES)
python tally.py
```

## Strategy

| Signal | Condition | Confidence |
|--------|-----------|------------|
| **YES** | Price < Weekly 0.5 Fib (Discount Zone) | 80% |
| **NO** | Price >= Weekly 0.5 Fib (Premium Zone) | 20% |

## Commands

| Command | Description |
|---------|-------------|
| `python quick_signal.py` | Get live binary signal |
| `python tally.py` | Record YES signal to backtest.csv |
| `python training/train_discount.py --data data/mnq_yfinance.csv --epochs 200 --batch 64 --yfinance` | Train discount model |

## Project Structure

```
MNQ-ME/
├── main.py                    # Main entry point
├── quick_signal.py            # Live signal generation
├── tally.py                   # Trade recording
├── requirements.txt           # Dependencies
├── config/
│   └── config.yaml            # Configuration
├── data/
│   ├── yfinance_fetcher.py    # Data pipeline
│   └── *_yfinance.csv         # OHLCV data
├── training/
│   ├── train_discount.py      # Model training
│   └── discount_labeler.py    # Labeling logic
├── models/
│   └── discount_detector.py   # Neural network model
└── results/
    ├── trained/               # Training results
    └── walkforward/           # Backtest results
```

## Data Source

- **Live Data**: Yahoo Finance (yfinance) - MNQ=F
- **Interval**: 1-hour bars for historical (8 months), 15-min for live
- **Correlation Assets**: ES, NQ, RTY, CL, GC, ZB, 6E

## Sponsor

[MicroCromwell.org](https://mycromwell.org) - Supporting quantitative trading research and development.

## License

Proprietary - For authorized use only.
# mnq-signal-cromwell
