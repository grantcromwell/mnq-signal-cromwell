# MNQ Discount Detection Trading System

## Executive Summary

A simplified binary trading system that detects "discount" opportunities in MNQ (Micro Nasdaq 100) futures by identifying price levels below the weekly 0.5 Fibonacci retracement level. The system generates YES/NO signals based on whether price is in the discount zone, with strict 80% confidence threshold and 3:1 risk/reward ratio targets.

---

## Problem Statement

### Current Issues
- **21% win rate** on real data indicates severe overfitting
- Complex VL-JEPA architecture with multiple data modalities (GAF images, correlation rings, order flow) makes debugging difficult
- 3:1 R/R framework has too many moving parts to validate
- Unclear if model is actually learning price action or memorizing patterns

### Solution
A **simple, measurable, tradeable** approach:
- Focus on single concept: **weekly 0.5 Fibonacci level as discount/premium divider**
- Binary YES/NO signals only
- 80% confidence threshold (stricter than current 65%)
- Success clearly defined as **reaching 3:1 target before stop**

---

## Core Concept: Discount vs Premium

### Fibonacci Calculation
```
Weekly High (WH)
Weekly Low (WL)
Weekly Range = WH - WL

0.5 Fib Level = WL + (Weekly Range × 0.5)
             = (Weekly High + Weekly Low) / 2

Discount Zone = Price < 0.5 Fib Level  ← BUY ZONE
Premium Zone  = Price ≥ 0.5 Fib Level  ← NO TRADE
```

### Visual Example
```
PREMIUM ZONE (Red)
███████████████████████
      0.5 Fib Level (17,600)
███████████████████████
DISCOUNT ZONE (Green)  ← TARGET BUY AREA
```

**Example:**
| Metric | Value |
|--------|-------|
| Week High | 17,800 |
| Week Low | 17,400 |
| 0.5 Fib | 17,600 |
| Discount Zone | Below 17,600 |
| Premium Zone | Above 17,600 |

---

## Trading Rules

### Entry Conditions (Signal YES)
1. Price is in **discount zone** (below weekly 0.5 Fib)
2. Model confidence ≥ **80%**
3. Time is within trading hours (3:00 AM - 3:00 PM EST)

### Exit Conditions
| Condition | Action |
|-----------|--------|
| Price hits Target (3:1 reward) | **WIN** - Close position |
| Price hits Stop (ATR-based) | **LOSS** - Close position |
| 15+ bars without hitting either | **FLAT** - Close without win/loss |

### Risk Management
| Parameter | Value | Notes |
|-----------|-------|-------|
| Risk per trade | 20% of portfolio | Aggressive but auto-adjusted |
| Max position | 12 contracts | Hard limit |
| Contract multiplier | 0.5 | MNQ micro |
| Stop loss | ATR-based | From config (currently 14-period) |
| Target | 3× Stop distance | Fixed R/R ratio |

---

## Labeling Strategy

### Success (Label = 1)
```
Entry:  Current bar close when price < 0.5 Fib
Stop:   Entry - (ATR × 0.5)  [aggressive but defined]
Target: Entry + (Stop Distance × 3)

SUCCESS if:
- Price hits Target before Stop within 10 bars
- Target achieved = 3:1 reward reached
```

### Failure (Label = 0)
```
FAILURE if:
- Price hits Stop before Target
- Price stays flat for 15+ bars
- OR: Price in premium zone (no trade)
```

### No Trade Zones
- Price ≥ 0.5 Fib (premium zone)
- Confidence < 80%
- Outside trading hours (3 AM - 3 PM EST)

---

## Feature Engineering

### Input Features (25 total)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `fib_distance` | Price distance to 0.5 Fib (pts) | (Price - Fib) / ATR |
| `fib_pct` | Price as % of Fib level | Price / Fib |
| `weekly_momentum` | 3-day price change | (Close - Close_3d) / Close_3d |
| `rsi_14` | 14-bar RSI | 0-100 scale |
| `atr_14` | Current ATR value | Raw value |
| `volume_ratio` | Current vol / 20-bar avg | Raw ratio |
| `hour_of_day` | Trading hour (3-15) | One-hot encoded |
| `day_of_week` | Monday=0 to Friday=4 | One-hot encoded |
| `bar_in_week` | Which bar of the week (0-52) | 0-1 normalized |
| `distance_to_weekly_high` | Pts to week high | (WH - Price) / ATR |
| `distance_to_weekly_low` | Pts to week low | (Price - WL) / ATR |

### Feature Vector Example
```
[0.5, 0.98, 0.02, 45, 61, 1.2, 0, 1, 0, 0.3, 0.5, ...]
[fib_dist, fib_pct, momentum, rsi, atr, vol, hour, day, ...]
```

---

## Model Architecture

### Simplified MLP Network

```
Input Layer (25 features)
        ↓
Dense (128) + LayerNorm + GELU + Dropout(0.2)
        ↓
Dense (64) + LayerNorm + GELU + Dropout(0.2)
        ↓
Dense (32) + LayerNorm + GELU
        ↓
Dense (2) + Softmax  →  [P(NO), P(YES)]
```

### Why This Architecture?
| Aspect | Rationale |
|--------|-----------|
| MLP over VL-JEPA | No images or text needed - pure price features |
| Small network | Avoid overfitting to limited data |
| LayerNorm | Stable training |
| Dropout 0.2 | Light regularization |
| 2-class output | Simple YES/NO decision |

---

## Training Pipeline

### Phase 1: Data Preparation
1. Load MNQ data (March 2025 to present)
2. Calculate weekly OHLC for each week
3. Compute 0.5 Fib level for each week
4. Label each 15-min bar (YES/NO/NO_TRADE)

### Phase 2: Feature Engineering
1. Calculate all 25 features for each bar
2. Remove bars with missing data
3. Shuffle and split: 60% train / 20% val / 20% test

### Phase 3: Training
```python
Hyperparameters:
- Batch size: 32
- Learning rate: 1e-4
- Weight decay: 0.01
- Early stopping: 10 epochs patience
- Max epochs: 100
- Loss: Binary cross-entropy
```

### Phase 4: Evaluation
```python
Metrics:
- Precision (P(YES|actual YES))
- Recall (P(actual YES|YES))
- F1 Score
- Accuracy
- Average R/R achieved
- Win rate on YES signals
```

---

## Success Criteria

| Metric | Target | Minimum |
|--------|--------|---------|
| Precision | ≥65% | 55% |
| Recall | ≥50% | 40% |
| Average R/R | ≥2.5:1 | 2.0:1 |
| Win Rate on YES | ≥50% | 40% |
| False Positive Rate | ≤35% | 45% |

---

## Position Sizing

### Auto-Risk Adjusted Formula
```
Portfolio Value = $100,000 (example)
Risk per Trade = 20% = $20,000

ATR = 61.4 points
Stop Distance = ATR × 0.5 = 30.7 points
Risk per Contract = 30.7 × 0.5 = $15.35

Position Size = Risk Amount / Risk per Contract
              = $20,000 / $15.35
              = 1,302 contracts

Max Position = 12 contracts (hard limit)
Final Position = min(1302, 12) = 12 contracts
```

### Risk per Contract
| Entry | Stop | Distance | Risk/Contract |
|-------|------|----------|---------------|
| 17,550 | 17,519 | 31 pts | $15.50 |
| 17,500 | 17,469 | 31 pts | $15.50 |
| 17,400 | 17,369 | 31 pts | $15.50 |

---

## Output Format

### Per-Bar Signal
```json
{
  "timestamp": "2025-03-15 09:45:00",
  "signal": "YES",
  "confidence": 0.85,
  "discount_level": 17600.0,
  "entry_price": 17550.0,
  "stop_loss": 17519.3,
  "target_3r": 17642.1,
  "risk_reward": "3.0:1",
  "position_size": 12,
  "reason": "Price 50pts below weekly 0.5 Fib, RSI=42, momentum positive",
  "expected_return": "$184.80",
  "max_risk": "$184.80"
}
```

### Daily Summary
```json
{
  "date": "2025-03-15",
  "total_signals": 3,
  "yes_signals": 2,
  "no_signals": 1,
  "wins": 1,
  "losses": 0,
  "flat": 1,
  "daily_pnl": "+$184.80",
  "win_rate": 1.0,
  "avg_confidence": 0.82
}
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `docs/discount_plan.md` | Create | This plan document |
| `data/fibonacci.py` | Create | Weekly 0.5 Fib calculator |
| `training/discount_labeler.py` | Create | Success/failure labeling |
| `models/discount_detector.py` | Create | Simplified binary classifier |
| `training/train_discount.py` | Create | Training pipeline |
| `config/discount_config.yaml` | Create | New configuration file |
| `main.py` | Modify | Add `--mode discount` option |
| `requirements.txt` | Modify | No new dependencies |

---

## Alpha Vantage Integration

### What's Included
- **API Key**: Already configured in `config/config.yaml`
- **News Sentiment**: Optional filter for market direction
- **Rate Limit**: 5 requests/minute (free tier)

### How It Helps
| Scenario | Sentiment | Action |
|----------|-----------|--------|
| Bullish news + Discount | +0.3 | Higher confidence YES |
| Bearish news + Discount | -0.3 | Lower confidence (may skip) |
| Neutral + Discount | 0.0 | Base confidence |

### Without API Key
- Sentiment defaults to 0 (neutral)
- System still works on price action alone

---

## Trading Hours (EST)

| Parameter | Value |
|-----------|-------|
| Start | 3:00 AM |
| End | 3:00 PM |
| Timezone | America/New_York |
| Weekend | Closed |

---

## Expected Improvements

| Metric | Current (VL-JEPA) | Target (Discount) |
|--------|-------------------|-------------------|
| Model complexity | High | **Low** |
| Confidence threshold | 65% | **80%** |
| Real data win rate | 21% | **50%+** |
| Clarity | Complex rules | **Simple** |
| Measurability | Ambiguous | **Clear** |
| Training time | Hours | **Minutes** |
| Inference speed | Slow (ViT) | **Fast** |

---

## Backtest Parameters

| Parameter | Value |
|-----------|-------|
| Date range | March 2025 - Present |
| Timeframe | 15-min bars |
| Initial capital | $100,000 |
| Risk per trade | 20% |
| Max position | 12 contracts |
| Commission | $0 (simulated) |

---

## Success Definition

**A trade is successful if:**
1. Signal was YES (confidence ≥80%)
2. Price reached 3:1 target before stop
3. Reward achieved = 3× risk

**A trade fails if:**
1. Signal was YES but price hit stop first
2. Signal was YES but price went flat for 15+ bars

---

## Next Steps

1. [ ] Create `docs/discount_plan.md`
2. [ ] Create `data/fibonacci.py` for Fib calculation
3. [ ] Create `training/discount_labeler.py` for labeling
4. [ ] Create `models/discount_detector.py` for the model
5. [ ] Create `training/train_discount.py` for training
6. [ ] Create `config/discount_config.yaml`
7. [ ] Update `main.py` with discount mode
8. [ ] Run backtest March 2025 - present
9. [ ] Evaluate and iterate

---

## Questions

1. **Stop loss width:** ATR-based (currently 14-period in config) - approved
2. **Time horizon:** 10 bars before declaring flat - approved
3. **API key:** Already in config - confirmed
4. **Position sizing:** Auto-risk adjusted with 20% risk - approved

---

*Document Version: 1.0*
*Created: January 15, 2026*
*Author: Claude (AI Trading System)*
