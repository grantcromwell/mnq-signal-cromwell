"""
IBKR Market Data Client for MNQ Trading Signal System

Uses ib-insync to connect to Interactive Brokers TWS/Gateway
and fetch real-time and historical futures data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from ib_insync import IB, Contract, Stock, Future, BarData
from ib_insync import util

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """OHLCV bar data structure"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        return cls(
            timestamp=pd.to_datetime(data["timestamp"]).to_pydatetime(),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            vwap=data.get("vwap"),
            count=data.get("count"),
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


class IBKRClient:
    """
    Async IBKR client for fetching futures data.
    Manages connection to TWS/Gateway and data retrieval.
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["ibkr"]

        self.host = self.config["host"]
        self.port = self.config["port"]
        self.client_id = self.config["client_id"]
        self.read_timeout = self.config.get("read_timeout", 60)

        self.ib = IB()
        self.connected = False

        # Contract cache to avoid repeated lookups
        self._contract_cache: Dict[str, Contract] = {}

    async def connect(self) -> bool:
        """Establish connection to IBKR TWS/Gateway"""
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.read_timeout,
            )
            self.connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def _get_contract(
        self,
        symbol: str,
        secType: str = "FUT",
        exchange: str = "CME",
        currency: str = "USD",
    ) -> Contract:
        """Get or create a contract object with caching"""
        cache_key = f"{symbol}_{secType}_{exchange}_{currency}"
        if cache_key in self._contract_cache:
            return self._contract_cache[cache_key]

        contract = Contract()
        contract.symbol = symbol
        contract.secType = secType
        contract.exchange = exchange
        contract.currency = currency

        self._contract_cache[cache_key] = contract
        return contract

    async def get_historical_bars(
        self,
        symbol: str,
        interval: str = "15m",
        duration: str = "14 D",
        bar_size: str = "1 min",
        end_datetime: Optional[datetime] = None,
        whatToShow: str = "MIDPOINT",
        useRTH: bool = True,
    ) -> List[OHLCV]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Contract symbol (e.g., "MNQ")
            interval: Output bar interval ("1m", "5m", "15m", "1h", "1d")
            duration: How far back ("1 D", "5 D", "14 D", "1 M")
            bar_size: Bar size setting ("1 min", "5 mins", "1 hour")
            end_datetime: End time for data
            whatToShow: Data type ("MIDPOINT", "TRADES", "BID", "ASK")
            useRTH: Use regular trading hours only

        Returns:
            List of OHLCV bars
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._get_contract(symbol)

        if end_datetime is None:
            end_datetime = datetime.now()

        try:
            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=2,  # Unix timestamp
            )

            ohlcv_list = []
            for bar in bars:
                ohlcv_list.append(
                    OHLCV(
                        timestamp=bar.date
                        if isinstance(bar.date, datetime)
                        else datetime.fromtimestamp(bar.date),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        vwap=bar.vwap,
                        count=bar.barCount,
                    )
                )

            logger.info(f"Fetched {len(ohlcv_list)} bars for {symbol} ({interval})")
            return ohlcv_list

        except Exception as e:
            logger.error(f"Error fetching historical bars for {symbol}: {e}")
            return []

    async def get_realtime_bars(
        self, symbol: str, interval: str = "15m", useRTH: bool = True
    ) -> asyncio.Queue:
        """
        Subscribe to real-time bars.

        Returns:
            asyncio.Queue receiving OHLCV updates
        """
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._get_contract(symbol)
        output_queue = asyncio.Queue()

        def on_bar_update(bars: List[BarData], has_new_bar: bool):
            for bar in bars:
                ohlcv = OHLCV(
                    timestamp=bar.date
                    if isinstance(bar.date, datetime)
                    else datetime.fromtimestamp(bar.date),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.vwap,
                    count=bar.barCount,
                )
                asyncio.create_task(output_queue.put(ohlcv))

        self.ib.reqRealTimeBars(
            contract=contract,
            barSize=interval,
            whatToShow="MIDPOINT",
            useRTH=useRTH,
            realTimeBarsCallback=on_bar_update,
        )

        return output_queue

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._get_contract(symbol)
        try:
            tickers = await self.ib.reqTickersAsync(contract)
            if tickers and tickers[0]:
                return tickers[0].close
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
        return None

    def bars_to_dataframe(self, bars: List[OHLCV]) -> pd.DataFrame:
        """Convert list of OHLCV bars to pandas DataFrame"""
        if not bars:
            return pd.DataFrame()

        data = [bar.to_dict() for bar in bars]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_contract_details(self, symbol: str) -> Optional[Dict]:
        """Get contract details and specifications"""
        if not self.connected:
            raise RuntimeError("Not connected to IBKR")

        contract = self._get_contract(symbol)
        try:
            details = self.ib.reqContractDetails(contract)
            if details:
                d = details[0]
                return {
                    "conId": d.contract.conId,
                    "symbol": d.contract.symbol,
                    "secType": d.contract.secType,
                    "exchange": d.contract.exchange,
                    "currency": d.contract.currency,
                    "multiplier": d.contract.multiplier,
                    "minTick": d.minTick,
                    "marketName": d.marketName,
                    "tradingHours": d.tradingHours,
                    "liquidHours": d.liquidHours,
                }
        except Exception as e:
            logger.error(f"Error getting contract details for {symbol}: {e}")
        return None


async def fetch_all_assets(
    client: IBKRClient, interval: str = "15m", days: int = 14
) -> Dict[str, List[OHLCV]]:
    """
    Fetch historical data for all correlation assets.

    Returns:
        Dict mapping symbol -> list of OHLCV bars
    """
    import yaml

    with open("../config/assets.yaml", "r") as f:
        assets_config = yaml.safe_load()

    all_data = {}

    # Combine all assets from both rings
    all_assets = assets_config.get("ring1_us_equity", []) + assets_config.get(
        "ring2_global", []
    )

    duration = f"{days} D"

    for asset in all_assets:
        symbol = asset["symbol"]
        bars = await client.get_historical_bars(
            symbol=symbol, interval=interval, duration=duration
        )
        all_data[symbol] = bars
        logger.info(f"Fetched {len(bars)} bars for {symbol}")

    return all_data


if __name__ == "__main__":

    async def test():
        client = IBKRClient()

        connected = await client.connect()
        if not connected:
            print("Failed to connect to IBKR")
            return

        # Test fetch MNQ data
        bars = await client.get_historical_bars(
            symbol="MNQ", interval="15m", duration="3 D"
        )

        if bars:
            df = client.bars_to_dataframe(bars)
            print(f"Fetched {len(bars)} bars")
            print(df.tail())

        await client.disconnect()

    asyncio.run(test())
