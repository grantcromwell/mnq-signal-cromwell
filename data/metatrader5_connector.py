"""
MetaTrader 5 Connector for CME Level 2 Data

Connects to MetaTrader 5 for real-time Level 2 order book data,
liquidation levels, and volume profile analysis.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MT5OrderBook:
    """Order book snapshot from MT5"""

    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]  # (price, volume)
    bid_count: int
    ask_count: int
    spread: float


@dataclass
class MT5Tick:
    """Single tick data from MT5"""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    flags: int


@dataclass
class MT5Level2Snapshot:
    """Complete Level 2 snapshot"""

    symbol: str
    timestamp: datetime
    order_book: MT5OrderBook
    recent_ticks: List[MT5Tick]
    daily_high: float
    daily_low: float
    daily_volume: float


@dataclass
class LiquidationLevel:
    """Detected liquidation level"""

    price: float
    direction: str  # "long_liquidation" or "short_liquidation"
    strength: float  # 0-1 confidence
    volume_bid: float
    volume_ask: float
    net_imbalance: float
    timestamp: datetime
    source: str  # "order_book", "price_action", "historical"


class MetaTrader5Connector:
    """
    MetaTrader 5 API connector for Level 2 data.

    MT5 provides:
    - Real-time tick data
    - Market depth (Level 2)
    - Order book snapshots
    - Historical tick data

    For CME futures, we use symbols like:
    - "MNQ" - Micro Nasdaq 100
    - "ES" - E-mini S&P 500
    - "NQ" - E-mini Nasdaq 100
    - "RTY" - E-mini Russell 2000
    """

    def __init__(
        self,
        api_key: str = "",
        server: str = "",
        login: int = 0,
        password: str = "",
        symbols: List[str] = None,
    ):
        self.api_key = api_key
        self.server = server
        self.login = login
        self.password = password
        self.symbols = symbols or ["MNQ", "ES", "NQ", "RTY"]
        self._connected = False
        self._order_book_cache = {}
        self._tick_buffer = defaultdict(list)
        self._liquidation_levels = []

    async def connect(self) -> bool:
        """
        Connect to MetaTrader 5 terminal.

        Returns:
            True if connected successfully
        """
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False

            if self.login > 0 and self.password:
                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server,
                )
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False

            self._connected = True
            logger.info(f"Connected to MT5 server: {self.server}")

            for symbol in self.symbols:
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Could not select symbol: {symbol}")
                else:
                    logger.info(f"Symbol selected: {symbol}")

            return True

        except ImportError:
            logger.error(
                "MetaTrader5 package not installed. Install with: pip install MetaTrader5"
            )
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MT5"""
        if self._connected:
            import MetaTrader5 as mt5

            mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    async def get_order_book(
        self, symbol: str, levels: int = 10
    ) -> Optional[MT5OrderBook]:
        """
        Get current order book for a symbol.

        Args:
            symbol: Trading symbol (e.g., "MNQ")
            levels: Number of price levels to retrieve

        Returns:
            MT5OrderBook or None if not available
        """
        if not self._connected:
            return None

        try:
            import MetaTrader5 as mt5

            book = mt5.market_book_get(symbol)
            if book is None:
                logger.warning(f"No order book available for {symbol}")
                return None

            bids = []
            asks = []
            for entry in book:
                if entry.type == 0:  # Bid
                    bids.append((entry.price, entry.volume))
                elif entry.type == 1:  # Ask
                    asks.append((entry.price, entry.volume))

            spread = 0
            if bids and asks:
                spread = asks[0][0] - bids[0][0]

            order_book = MT5OrderBook(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bids=bids[:levels],
                asks=asks[:levels],
                bid_count=len(bids),
                ask_count=len(asks),
                spread=spread,
            )

            self._order_book_cache[symbol] = order_book
            return order_book

        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    async def get_ticks(
        self,
        symbol: str,
        from_date: datetime = None,
        to_date: datetime = None,
        count: int = 1000,
    ) -> List[MT5Tick]:
        """
        Get historical tick data.

        Args:
            symbol: Trading symbol
            from_date: Start datetime
            to_date: End datetime
            count: Maximum number of ticks

        Returns:
            List of MT5Tick objects
        """
        if not self._connected:
            return []

        try:
            import MetaTrader5 as mt5

            if from_date is None:
                from_date = datetime.utcnow() - timedelta(hours=1)
            if to_date is None:
                to_date = datetime.utcnow()

            ticks = mt5.copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)

            if ticks is None or len(ticks) == 0:
                return []

            tick_list = []
            for tick in ticks[-count:]:
                tick_list.append(
                    MT5Tick(
                        symbol=symbol,
                        timestamp=pd.to_datetime(tick["time"], unit="s"),
                        bid=tick["bid"],
                        ask=tick["ask"],
                        last=tick["last"],
                        volume=tick["volume"],
                        flags=tick["flags"],
                    )
                )

            self._tick_buffer[symbol].extend(tick_list[-10000:])
            return tick_list

        except Exception as e:
            logger.error(f"Error getting ticks for {symbol}: {e}")
            return []

    async def get_daily_bars(
        self,
        symbol: str,
        from_date: datetime = None,
        to_date: datetime = None,
        timeframe: int = mt5.TIMEFRAME_M15,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Trading symbol
            from_date: Start datetime
            to_date: End datetime
            timeframe: MT5 timeframe constant

        Returns:
            DataFrame with OHLCV data
        """
        if not self._connected:
            return pd.DataFrame()

        try:
            import MetaTrader5 as mt5

            if from_date is None:
                from_date = datetime.utcnow() - timedelta(days=30)
            if to_date is None:
                to_date = datetime.utcnow()

            rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

            if rates is None or len(rates) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting rates for {symbol}: {e}")
            return pd.DataFrame()

    async def stream_order_book(
        self,
        symbol: str,
        callback,
        interval: float = 0.1,
    ):
        """
        Stream order book updates.

        Args:
            symbol: Trading symbol
            callback: Async callback function
            interval: Update interval in seconds
        """
        if not self._connected:
            return

        while self._connected:
            book = await self.get_order_book(symbol)
            if book:
                await callback(book)
            await asyncio.sleep(interval)

    async def stream_ticks(
        self,
        symbol: str,
        callback,
        interval: float = 0.05,
    ):
        """
        Stream tick updates.

        Args:
            symbol: Trading symbol
            callback: Async callback function
            interval: Update interval in seconds
        """
        if not self._connected:
            return

        last_count = 0
        while self._connected:
            ticks = await self.get_ticks(symbol, count=100)
            new_ticks = ticks[last_count:]
            for tick in new_ticks:
                await callback(tick)
            last_count = len(ticks)
            await asyncio.sleep(interval)


class MT5Level2Analyzer:
    """
    Level 2 analysis using MT5 data.
    """

    def __init__(self, connector: MetaTrader5Connector):
        self.connector = connector
        self._tick_history = defaultdict(list)
        self._volume_profile = {}

    async def get_level2_snapshot(self, symbol: str) -> Optional[MT5Level2Snapshot]:
        """
        Get complete Level 2 snapshot.
        """
        order_book = await self.connector.get_order_book(symbol)
        if order_book is None:
            return None

        now = datetime.utcnow()
        ticks = self._tick_history.get(symbol, [])[-100:]

        df = await self.connector.get_daily_bars(symbol)
        daily_high = (
            df["high"].max()
            if not df.empty
            else order_book.bids[0][0]
            if order_book.bids
            else 0
        )
        daily_low = (
            df["low"].min()
            if not df.empty
            else order_book.asks[0][0]
            if order_book.asks
            else 0
        )
        daily_volume = df["volume"].sum() if "volume" in df.columns else 0

        return MT5Level2Snapshot(
            symbol=symbol,
            timestamp=now,
            order_book=order_book,
            recent_ticks=ticks,
            daily_high=daily_high,
            daily_low=daily_low,
            daily_volume=daily_volume,
        )

    def detect_liquidation_levels(
        self,
        order_book: MT5OrderBook,
        current_price: float,
        daily_high: float,
        daily_low: float,
        tick_history: List[MT5Tick] = None,
    ) -> List[LiquidationLevel]:
        """
        Detect liquidation levels from order book.

        Long liquidation: Price near highs with weak bids
        Short liquidation: Price near lows with weak asks
        """
        levels = []

        total_bid_vol = sum(v for _, v in order_book.bids)
        total_ask_vol = sum(v for _, v in order_book.asks)

        liquidation_threshold = 0.015  # 1.5% from high/low

        long_liq_zone_high = daily_high * (1 - liquidation_threshold)
        long_liq_zone_low = daily_high

        bid_vol_in_liq_zone = sum(
            v
            for p, v in order_book.bids
            if long_liq_zone_high <= p <= long_liq_zone_low
        )
        if total_bid_vol > 0:
            weak_bid_ratio = 1 - (bid_vol_in_liq_zone / total_bid_vol)
            if weak_bid_ratio > 0.6 and bid_vol_in_liq_zone < total_bid_vol * 0.2:
                levels.append(
                    LiquidationLevel(
                        price=daily_high,
                        direction="long_liquidation",
                        strength=min(1.0, weak_bid_ratio * 1.2),
                        volume_bid=bid_vol_in_liq_zone,
                        volume_ask=0,
                        net_imbalance=-weak_bid_ratio,
                        timestamp=order_book.timestamp,
                        source="order_book",
                    )
                )

        short_liq_zone_high = daily_low
        short_liq_zone_low = daily_low * (1 + liquidation_threshold)

        ask_vol_in_liq_zone = sum(
            v
            for p, v in order_book.asks
            if short_liq_zone_high <= p <= short_liq_zone_low
        )
        if total_ask_vol > 0:
            weak_ask_ratio = 1 - (ask_vol_in_liq_zone / total_ask_vol)
            if weak_ask_ratio > 0.6 and ask_vol_in_liq_zone < total_ask_vol * 0.2:
                levels.append(
                    LiquidationLevel(
                        price=daily_low,
                        direction="short_liquidation",
                        strength=min(1.0, weak_ask_ratio * 1.2),
                        volume_bid=0,
                        volume_ask=ask_vol_in_liq_zone,
                        net_imbalance=weak_ask_ratio,
                        timestamp=order_book.timestamp,
                        source="order_book",
                    )
                )

        if tick_history and len(tick_history) > 50:
            levels.extend(
                self._detect_historical_liquidation(tick_history, current_price)
            )

        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:10]

    def _detect_historical_liquidation(
        self,
        ticks: List[MT5Tick],
        current_price: float,
    ) -> List[LiquidationLevel]:
        """Detect historical liquidation levels from tick data"""
        levels = []

        if len(ticks) < 50:
            return levels

        highs = [t.last for t in ticks[-100:] if t.last > 0]
        lows = [t.last for t in ticks[-100:] if t.last > 0]

        if not highs or not lows:
            return levels

        recent_high = max(highs)
        recent_low = min(lows)

        price_levels = {}
        for tick in ticks[-1000:]:
            price_rounded = round(tick.last, 0)
            if price_rounded not in price_levels:
                price_levels[price_rounded] = {
                    "volumes": [],
                    "tick_count": 0,
                    "last_time": tick.timestamp,
                }
            price_levels[price_rounded]["volumes"].append(tick.volume)
            price_levels[price_rounded]["tick_count"] += 1

        for price, data in price_levels.items():
            avg_vol = np.mean(data["volumes"])
            vol_std = np.std(data["volumes"])
            if vol_std > 0:
                z_score = (data["volumes"][-1] - avg_vol) / vol_std
                if z_score < -1.5 and abs(price - recent_high) / recent_high < 0.02:
                    levels.append(
                        LiquidationLevel(
                            price=float(price),
                            direction="long_liquidation",
                            strength=min(1.0, abs(z_score) / 3),
                            volume_bid=0,
                            volume_ask=0,
                            net_imbalance=z_score / 3,
                            timestamp=data["last_time"],
                            source="price_action",
                        )
                    )

        return levels

    def compute_volume_profile(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        volumes: pd.Series,
        tick_history: List[MT5Tick] = None,
        price_bins: int = 50,
    ) -> Dict[str, Any]:
        """
        Compute volume profile with tick data augmentation.
        """
        price_range = highs.max() - lows.min()
        bin_size = price_range / price_bins

        price_centers = np.linspace(
            lows.min() + bin_size / 2,
            highs.max() - bin_size / 2,
            price_bins,
        )

        volume_at_price = np.zeros(price_bins)
        buy_volume = np.zeros(price_bins)
        sell_volume = np.zeros(price_bins)

        for i in range(len(closes)):
            high_idx = min(
                int((highs.iloc[i] - lows.min()) / bin_size),
                price_bins - 1,
            )
            low_idx = min(
                int((lows.iloc[i] - lows.min()) / bin_size),
                price_bins - 1,
            )

            avg_volume = volumes.iloc[i] / max(high_idx - low_idx + 1, 1)

            if i > 0 and closes.iloc[i] > closes.iloc[i - 1]:
                buy_volume[low_idx : high_idx + 1] += avg_volume
            elif i > 0:
                sell_volume[low_idx : high_idx + 1] += avg_volume

            volume_at_price[low_idx : high_idx + 1] += avg_volume

        if tick_history:
            tick_volumes = [t.volume for t in tick_history[-1000:] if t.volume > 0]
            if tick_volumes:
                avg_tick_vol = np.mean(tick_volumes)
                for tick in tick_history[-1000:]:
                    if tick.last > 0:
                        price_idx = min(
                            int((tick.last - lows.min()) / bin_size),
                            price_bins - 1,
                        )
                        volume_at_price[price_idx] += avg_tick_vol * 0.1

        poc_idx = np.argmax(volume_at_price)
        poc = price_centers[poc_idx]
        poc_volume = volume_at_price[poc_idx]

        cumsum = np.cumsum(volume_at_price)
        val_area_low_idx = np.searchsorted(cumsum, (1 - 0.70) / 2)
        val_area_high_idx = np.searchsorted(cumsum, (1 + 0.70) / 2)

        value_area_high = price_centers[min(val_area_high_idx, len(price_centers) - 1)]
        value_area_low = price_centers[max(val_area_low_idx, 0)]

        vol_above_poc = volume_at_price[price_centers > poc].sum()
        vol_below_poc = volume_at_price[price_centers < poc].sum()

        return {
            "price_levels": price_centers,
            "volume_at_price": volume_at_price,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "poc": poc,
            "poc_volume": poc_volume,
            "value_area_high": value_area_high,
            "value_area_low": value_area_low,
            "volume_asymmetry": vol_above_poc / max(vol_below_poc, 0.001),
            "total_volume": volume_at_price.sum(),
        }

    def compute_order_flow_metrics(
        self,
        order_book: MT5OrderBook,
        ticks: List[MT5Tick],
    ) -> Dict[str, float]:
        """
        Compute order flow and delta metrics.
        """
        total_bid_vol = sum(v for _, v in order_book.bids)
        total_ask_vol = sum(v for _, v in order_book.asks)

        cum_delta = total_bid_vol - total_ask_vol

        order_imbalance = (
            (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
            if (total_bid_vol + total_ask_vol) > 0
            else 0
        )

        bid_ask_imbalance = order_imbalance

        delta_ratio = 0
        total_volume = sum(t.volume for t in ticks) if ticks else 0
        if total_volume > 0:
            delta_ratio = cum_delta / total_volume

        absorption_score = 0
        if len(ticks) > 10:
            recent_ticks = ticks[-10:]
            recent_delta = sum(
                (t.last - (t.bid + t.ask) / 2) * t.volume for t in recent_ticks
            )
            if total_volume > 0:
                absorption_score = max(0, min(1, 0.5 - recent_delta / total_volume))

        return {
            "cum_delta": cum_delta,
            "delta_ratio": delta_ratio,
            "order_imbalance": order_imbalance,
            "bid_ask_spread": order_book.spread,
            "bid_ask_imbalance": bid_ask_imbalance,
            "absorption_score": absorption_score,
            "total_bid_volume": total_bid_vol,
            "total_ask_volume": total_ask_vol,
            "bid_count": order_book.bid_count,
            "ask_count": order_book.ask_count,
        }

    def create_order_book_features(
        self,
        volume_profile: Dict,
        order_flow: Dict,
        liquidation_levels: List[LiquidationLevel],
        current_price: float,
    ) -> np.ndarray:
        """
        Create feature vector from order book analysis.
        """
        features = []

        features.append(
            (volume_profile.get("poc", current_price) - current_price) / current_price
        )
        features.append(volume_profile.get("poc_volume", 0))
        features.append(volume_profile.get("volume_asymmetry", 1))

        features.append(order_flow.get("cum_delta", 0))
        features.append(order_flow.get("delta_ratio", 0))
        features.append(order_flow.get("order_imbalance", 0))
        features.append(order_flow.get("bid_ask_spread", 0))
        features.append(order_flow.get("absorption_score", 0))

        long_liq_count = sum(
            1 for l in liquidation_levels if l.direction == "long_liquidation"
        )
        short_liq_count = sum(
            1 for l in liquidation_levels if l.direction == "short_liquidation"
        )
        avg_liq_strength = (
            np.mean([l.strength for l in liquidation_levels])
            if liquidation_levels
            else 0
        )

        features.extend(
            [
                long_liq_count / 10,
                short_liq_count / 10,
                avg_liq_strength,
            ]
        )

        for i in range(5):
            if i < len(volume_profile.get("volume_at_price", [])):
                features.append(volume_profile["volume_at_price"][i] * 10)
            else:
                features.append(0)

        features.extend([0] * (30 - len(features)))
        return np.array(features[:30], dtype=np.float32)


def get_mt5_data(
    symbol: str = "MNQ",
    api_key: str = "M29EUYU9QD0LZEMN",
    from_date: datetime = None,
    to_date: datetime = None,
) -> Dict[str, Any]:
    """
    Main function to get MT5 data for backtesting.
    """
    import MetaTrader5 as mt5

    if not mt5.initialize():
        return {"error": "MT5 initialization failed"}

    connector = MetaTrader5Connector(
        api_key=api_key,
        symbols=[symbol],
    )

    if from_date is None:
        from_date = datetime(2025, 3, 1)
    if to_date is None:
        to_date = datetime.utcnow()

    df = pd.DataFrame()
    try:
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, from_date, to_date)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
    except Exception as e:
        logger.error(f"Error getting MT5 data: {e}")

    mt5.shutdown()
    return df


if __name__ == "__main__":
    df = get_mt5_data("MNQ", from_date=datetime(2025, 3, 1))
    if not df.empty:
        print(f"Loaded {len(df)} bars from MT5")
        print(df.head())
    else:
        print("No data loaded - MT5 may not be connected")
