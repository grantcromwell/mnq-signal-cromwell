"""
Volatility-Based Position Sizer

Calculates position sizes based on:
- Target portfolio volatility (e.g., 2% per trade)
- Current ATR (volatility measure)
- Kelly criterion for risk management
- Model confidence adjustment

Ensures proper risk management for 3:1 R/R strategy.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Position sizing result"""

    position_size: int
    contracts: int
    risk_amount: float
    reward_amount: float
    risk_per_contract: float
    reward_per_contract: float
    kelly_fraction: float
    confidence_multiplier: float
    final_size: int

    def to_dict(self) -> Dict:
        return {
            "position_size": self.position_size,
            "contracts": self.contracts,
            "risk_amount": self.risk_amount,
            "reward_amount": self.reward_amount,
            "risk_per_contract": self.risk_per_contract,
            "reward_per_contract": self.reward_per_contract,
            "kelly_fraction": self.kelly_fraction,
            "confidence_multiplier": self.confidence_multiplier,
            "final_size": self.final_size,
        }


class VolatilityPositionSizer:
    """
    Position sizing based on volatility and Kelly criterion.

    Formula:
    1. Calculate dollar risk per contract: (Entry - Stop) * Multiplier
    2. Calculate target portfolio risk: Portfolio Value * Volatility Target
    3. Base contracts = Target Risk / Dollar Risk per Contract
    4. Apply Kelly fraction (half-Kelly for safety)
    5. Adjust by model confidence
    6. Round to nearest integer within min/max bounds
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load()["position_sizing"]

        self.volatility_target = self.config["volatility_target"]
        self.atr_period = self.config["atr_period"]
        self.kelly_fraction = self.config["kelly_fraction"]
        self.max_contracts = self.config["max_contracts"]
        self.min_contracts = self.config["min_contracts"]

        self.portfolio_value = 100000.0
        self.mnq_multiplier = 0.5

        logger.info(
            f"PositionSizer initialized: "
            f"Vol Target: {self.volatility_target:.1%}, "
            f"Kelly: {self.kelly_fraction}, "
            f"MNQ Multiplier: ${self.mnq_multiplier}"
        )

    def set_portfolio_value(self, value: float):
        """Set current portfolio value for sizing calculations"""
        self.portfolio_value = value
        logger.info(f"Portfolio value set to: ${value:,.2f}")

    def calculate(
        self,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        atr: float,
        direction: str = "long",
    ) -> Tuple[int, float, float]:
        """
        Calculate position size for a trade.

        Args:
            confidence: Model confidence (0.0 - 1.0)
            entry_price: Planned entry price
            stop_loss: Stop loss price
            atr: Current ATR value
            direction: "long" or "short"

        Returns:
            Tuple of (position_size, risk_amount, reward_amount)
        """
        risk_per_point = abs(entry_price - stop_loss)
        risk_per_contract = risk_per_point * self.mnq_multiplier

        if risk_per_contract <= 0:
            logger.warning("Invalid risk calculation, using default")
            return 0, 0.0, 0.0

        target_risk_dollars = self.portfolio_value * self.volatility_target

        base_contracts = target_risk_dollars / risk_per_contract

        kelly_adjusted = base_contracts * self.kelly_fraction

        confidence_multiplier = self._confidence_multiplier(confidence)

        final_contracts = kelly_adjusted * confidence_multiplier

        final_contracts = self._apply_bounds(final_contracts)

        position_size = int(round(final_contracts))
        position_size = max(0, position_size)

        risk_amount = position_size * risk_per_contract
        reward_3r = risk_per_contract * 3
        reward_amount = position_size * reward_3r

        logger.debug(
            f"Position sizing: confidence={confidence:.2f}, "
            f"base={base_contracts:.2f}, kelly={kelly_adjusted:.2f}, "
            f"final={position_size}, risk=${risk_amount:.2f}"
        )

        return position_size, risk_amount, reward_amount

    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Adjust position size based on model confidence.

        Higher confidence = larger positions
        Confidence < 0.5 = minimal sizing

        Args:
            confidence: Model confidence (0.0 - 1.0)

        Returns:
            Multiplier for position sizing
        """
        if confidence < 0.5:
            return 0.0
        elif confidence < 0.7:
            return 0.5
        elif confidence < 0.8:
            return 0.75
        elif confidence < 0.9:
            return 1.0
        else:
            return 1.25

    def _apply_bounds(self, contracts: float) -> float:
        """
        Apply minimum and maximum position bounds.

        Args:
            contracts: Calculated number of contracts

        Returns:
            Bounded number of contracts
        """
        return max(self.min_contracts, min(self.max_contracts, contracts))

    def calculate_3r_size(
        self,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        target_3r: float,
        atr: float,
    ) -> PositionSizeResult:
        """
        Calculate position size with explicit 3R target.

        Args:
            confidence: Model confidence
            entry_price: Trade entry price
            stop_loss: Stop loss price
            target_3r: 3R target price
            atr: Current ATR

        Returns:
            PositionSizeResult object
        """
        risk_per_point = abs(entry_price - stop_loss)
        reward_per_point = abs(target_3r - entry_price)

        risk_per_contract = risk_per_point * self.mnq_multiplier
        reward_per_contract = reward_per_point * self.mnq_multiplier

        if risk_per_contract <= 0:
            return PositionSizeResult(
                position_size=0,
                contracts=0,
                risk_amount=0,
                reward_amount=0,
                risk_per_contract=0,
                reward_per_contract=0,
                kelly_fraction=0,
                confidence_multiplier=0,
                final_size=0,
            )

        target_risk_dollars = self.portfolio_value * self.volatility_target
        base_contracts = target_risk_dollars / risk_per_contract

        kelly_fraction_used = self.kelly_fraction * min(confidence * 1.5, 1.0)
        kelly_adjusted = base_contracts * kelly_fraction_used

        confidence_mult = self._confidence_multiplier(confidence)
        final_contracts = kelly_adjusted * confidence_mult

        final_contracts = self._apply_bounds(final_contracts)
        position_size = int(round(final_contracts))

        risk_amount = position_size * risk_per_contract
        reward_amount = position_size * reward_per_contract

        return PositionSizeResult(
            position_size=position_size,
            contracts=position_size,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_per_contract=risk_per_contract,
            reward_per_contract=reward_per_contract,
            kelly_fraction=kelly_fraction_used,
            confidence_multiplier=confidence_mult,
            final_size=position_size,
        )

    def expected_value(
        self, win_rate: float, avg_win: float, avg_loss: float, num_trades: int
    ) -> Dict[str, float]:
        """
        Calculate expected value of the position sizing strategy.

        Args:
            win_rate: Historical win rate (0.0 - 1.0)
            avg_win: Average winning trade profit
            avg_loss: Average losing trade loss (positive number)
            num_trades: Number of trades

        Returns:
            Dict of expected value metrics
        """
        expected_profit_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        total_expected = expected_profit_per_trade * num_trades

        kelly_optimal = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)

        return {
            "expected_profit_per_trade": expected_profit_per_trade,
            "total_expected_profit": total_expected,
            "kelly_optimal_fraction": max(0, kelly_optimal),
            "half_kelly_recommended": max(0, kelly_optimal / 2),
            "edge": expected_profit_per_trade / avg_loss if avg_loss > 0 else 0,
        }


class AdaptivePositionSizer(VolatilityPositionSizer):
    """
    Adaptive position sizer that adjusts based on:
    - Recent performance
    - Drawdown state
    - Market volatility regime
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        super().__init__(config_path)

        self.drawdown_threshold = 0.10
        self.high_vol_multiplier = 0.75
        self.low_vol_multiplier = 1.25
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3

    def calculate(
        self,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        atr: float,
        direction: str = "long",
        current_drawdown: float = 0.0,
        recent_performance: Optional[Dict] = None,
    ) -> Tuple[int, float, float]:
        """
        Calculate position size with adaptive adjustments.

        Args:
            confidence: Model confidence
            entry_price: Entry price
            stop_loss: Stop loss
            atr: Current ATR
            direction: Trade direction
            current_drawdown: Current portfolio drawdown
            recent_performance: Dict with recent trade results

        Returns:
            Tuple of (position_size, risk_amount, reward_amount)
        """
        if recent_performance:
            self._update_consecutive_losses(recent_performance)

        volatility_multiplier = self._volatility_multiplier(atr)
        drawdown_multiplier = self._drawdown_multiplier(current_drawdown)
        heat_multiplier = self._heat_multiplier()

        base_size, risk, reward = super().calculate(
            confidence, entry_price, stop_loss, atr, direction
        )

        adjusted_size = int(
            base_size * volatility_multiplier * drawdown_multiplier * heat_multiplier
        )

        adjusted_size = max(0, adjusted_size)

        adjusted_risk = adjusted_size * (
            abs(entry_price - stop_loss) * self.mnq_multiplier
        )

        return adjusted_size, adjusted_risk, adjusted_risk * 3

    def _update_consecutive_losses(self, recent_performance: Dict):
        """Update consecutive loss counter"""
        last_trades = recent_performance.get("last_trades", [])
        if len(last_trades) >= self.max_consecutive_losses:
            recent = last_trades[-self.max_consecutive_losses :]
            if all(t < 0 for t in recent):
                self.consecutive_losses = min(
                    self.consecutive_losses + 1, self.max_consecutive_losses
                )
            else:
                self.consecutive_losses = 0

    def _volatility_multiplier(self, atr: float) -> float:
        """Adjust for volatility regime"""
        atr_normalized = atr / 50.0

        if atr_normalized > 1.5:
            return self.high_vol_multiplier
        elif atr_normalized < 0.7:
            return self.low_vol_multiplier
        return 1.0

    def _drawdown_multiplier(self, drawdown: float) -> float:
        """Reduce sizing in drawdown"""
        if drawdown > self.drawdown_threshold:
            return 0.5
        return 1.0

    def _heat_multiplier(self) -> float:
        """Reduce sizing after consecutive losses"""
        if self.consecutive_losses >= self.max_consecutive_losses:
            return 0.5
        elif self.consecutive_losses >= 2:
            return 0.75
        return 1.0


if __name__ == "__main__":
    sizer = VolatilityPositionSizer()

    test_cases = [
        (0.95, 17000, 16950, 50),
        (0.85, 17000, 16950, 50),
        (0.75, 17000, 16950, 50),
        (0.65, 17000, 16950, 50),
        (0.45, 17000, 16950, 50),
    ]

    print("Position Sizing Examples:")
    print("-" * 60)

    for confidence, entry, sl, atr in test_cases:
        size, risk, reward = sizer.calculate(confidence, entry, sl, atr, "long")
        print(
            f"Conf: {confidence:.0%} | Entry: {entry} | SL: {sl} | "
            f"ATR: {atr} | Size: {size} | Risk: ${risk:.0f} | Reward: ${reward:.0f}"
        )

    print("-" * 60)

    ev = sizer.expected_value(win_rate=0.55, avg_win=150, avg_loss=50, num_trades=100)

    print(f"\nExpected Value Analysis:")
    for key, value in ev.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
