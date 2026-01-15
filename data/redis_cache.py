"""
Redis TimeSeries Cache for MNQ Trading Signal System

Stores 14-day rolling window of OHLCV data for MNQ and correlation assets.
Uses Redis TimeSeries module for efficient time-series storage and retrieval.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd

import redis
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Redis cache configuration"""

    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    key_prefix: str = "mnq:signal:"
    ttl_days: int = 14
    max_points_per_key: int = 100000


class RedisTimeSeriesCache:
    """
    Redis TimeSeries cache for storing and retrieving OHLCV data.
    Maintains 14-day rolling window of price data.
    """

    def __init__(self, config_path: str = "../config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["redis"]

        self.host = self.config["host"]
        self.port = self.config["port"]
        self.password = self.config.get("password")
        self.db = self.config.get("db", 0)
        self.key_prefix = self.config["key_prefix"]
        self.ttl_days = self.config["ttl_days"]
        self.max_points = self.config.get("max_points_per_key", 100000)

        self.client = self._connect()
        self._ensure_ts_module()

    def _connect(self) -> redis.Redis:
        """Create Redis connection"""
        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
            )
            client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return client
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _ensure_ts_module(self):
        """Ensure Redis TimeSeries module is loaded"""
        try:
            info = self.client.info("modules")
            ts_loaded = any(
                "timeseries" in str(info.get("module", {})).lower()
                for info in [info]
                if info
            )
            if not ts_loaded:
                logger.warning("Redis TimeSeries module may not be loaded")
        except Exception as e:
            logger.warning(f"Could not check modules: {e}")

    def _ts_key(self, symbol: str, interval: str) -> str:
        """Generate TimeSeries key for a symbol/interval"""
        return f"{self.key_prefix}ts:{symbol}:{interval}"

    def _meta_key(self, symbol: str, interval: str) -> str:
        """Generate metadata key"""
        return f"{self.key_prefix}meta:{symbol}:{interval}"

    def _corr_key(self, symbol: str) -> str:
        """Generate correlation data key"""
        return f"{self.key_prefix}corr:{symbol}"

    def create_time_series(
        self, symbol: str, interval: str, retention_ms: Optional[int] = None
    ) -> bool:
        """
        Create a TimeSeries key for a symbol/interval.

        Args:
            symbol: Trading symbol
            interval: Bar interval (1m, 5m, 15m, 1h, 1d)
            retention_ms: Retention period in milliseconds (default: 14 days)

        Returns:
            True if created or already exists
        """
        key = self._ts_key(symbol, interval)

        if retention_ms is None:
            retention_ms = self.ttl_days * 24 * 60 * 60 * 1000

        try:
            self.client.ts().create(
                key,
                retention_msecs=retention_ms,
                labels={
                    "symbol": symbol,
                    "interval": interval,
                    "created": datetime.now().isoformat(),
                },
            )
            logger.info(f"Created TimeSeries: {key}")
            return True
        except redis.ResponseError as e:
            if "TSDB: key already exists" in str(e):
                logger.debug(f"TimeSeries already exists: {key}")
                return True
            logger.error(f"Error creating TimeSeries {key}: {e}")
            return False

    def add_bar(
        self,
        symbol: str,
        interval: str,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        vwap: Optional[float] = None,
    ) -> bool:
        """
        Add a single bar to the TimeSeries.

        Args:
            symbol: Trading symbol
            interval: Bar interval
            timestamp: Bar timestamp
            open, high, low, close: OHLC prices
            volume: Trading volume
            vwap: Volume-weighted average price

        Returns:
            True if successful
        """
        key = self._ts_key(symbol, interval)
        ts_ms = int(timestamp.timestamp() * 1000)

        try:
            self.client.ts().add(
                key,
                ts_ms,
                [open, high, low, close, volume, vwap or close],
                retention_msecs=self.ttl_days * 24 * 60 * 60 * 1000,
            )
            return True
        except Exception as e:
            logger.error(f"Error adding bar to {key}: {e}")
            return False

    def add_bars_batch(
        self, symbol: str, interval: str, bars: List[Dict[str, Any]]
    ) -> int:
        """
        Add multiple bars in a batch.

        Args:
            symbol: Trading symbol
            interval: Bar interval
            bars: List of bar dictionaries

        Returns:
            Number of bars added
        """
        if not bars:
            return 0

        key = self._ts_key(symbol, interval)

        try:
            data = []
            for bar in bars:
                ts_ms = int(pd.to_datetime(bar["timestamp"]).timestamp() * 1000)
                data.append(
                    (
                        ts_ms,
                        [
                            bar["open"],
                            bar["high"],
                            bar["low"],
                            bar["close"],
                            bar["volume"],
                            bar.get("vwap", bar["close"]),
                        ],
                    )
                )

            self.client.ts().add(key, data)
            logger.info(f"Added {len(bars)} bars to {key}")
            return len(bars)
        except Exception as e:
            logger.error(f"Error batch adding bars to {key}: {e}")
            return 0

    def get_bars(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        count: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Retrieve bars from the TimeSeries.

        Args:
            symbol: Trading symbol
            interval: Bar interval
            start_time: Start of time range
            end_time: End of time range
            count: Maximum number of bars to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        key = self._ts_key(symbol, interval)

        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(days=self.ttl_days)

            start_ms = int(start_time.timestamp() * 1000)

            if end_time is None:
                end_ms = "+"  # Now
            else:
                end_ms = int(end_time.timestamp() * 1000)

            if count:
                data = self.client.ts().range(
                    key,
                    start_ms,
                    end_ms,
                    count=count,
                    aggregation_type="last",
                    bucket_duration_msec=1,
                )
            else:
                data = self.client.ts().range(key, start_ms, end_ms)

            if not data:
                return pd.DataFrame()

            records = []
            for ts, values in data:
                records.append(
                    {
                        "timestamp": datetime.fromtimestamp(ts / 1000),
                        "open": values[0],
                        "high": values[1],
                        "low": values[2],
                        "close": values[3],
                        "volume": int(values[4]),
                        "vwap": values[5],
                    }
                )

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error retrieving bars from {key}: {e}")
            return pd.DataFrame()

    def get_latest_bar(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Get the most recent bar for a symbol"""
        key = self._ts_key(symbol, interval)

        try:
            data = self.client.ts().get(key)
            if data:
                ts, values = data
                return {
                    "timestamp": datetime.fromtimestamp(ts / 1000),
                    "open": values[0],
                    "high": values[1],
                    "low": values[2],
                    "close": values[3],
                    "volume": int(values[4]),
                    "vwap": values[5],
                }
        except Exception as e:
            logger.error(f"Error getting latest bar from {key}: {e}")
        return None

    def store_correlation_matrix(
        self, symbol: str, correlation_data: Dict[str, Any]
    ) -> bool:
        """
        Store correlation matrix for an asset.

        Args:
            symbol: Trading symbol
            correlation_data: Correlation matrix and metadata

        Returns:
            True if successful
        """
        key = self._corr_key(symbol)

        try:
            self.client.setex(
                key,
                self.ttl_days * 24 * 60 * 60,
                json.dumps(correlation_data, default=str),
            )
            return True
        except Exception as e:
            logger.error(f"Error storing correlation: {e}")
            return False

    def get_correlation_matrix(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve correlation matrix for an asset"""
        key = self._corr_key(symbol)

        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving correlation: {e}")
        return None

    def store_rolling_features(self, symbol: str, features: Dict[str, Any]) -> bool:
        """Store computed features for quick retrieval"""
        key = f"{self.key_prefix}feat:{symbol}"

        try:
            self.client.setex(
                key,
                60 * 60,  # 1 hour TTL
                json.dumps(features, default=str),
            )
            return True
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False

    def get_rolling_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored features"""
        key = f"{self.key_prefix}feat:{symbol}"

        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
        return None

    def delete_symbol(self, symbol: str, interval: Optional[str] = None):
        """Delete cached data for a symbol"""
        if interval:
            key = self._ts_key(symbol, interval)
            try:
                self.client.delete(key)
                logger.info(f"Deleted {key}")
            except Exception as e:
                logger.error(f"Error deleting {key}: {e}")
        else:
            pattern = f"{self.key_prefix}*{symbol}*"
            try:
                for key in self.client.scan_iter(match=pattern):
                    self.client.delete(key)
                    logger.info(f"Deleted {key}")
            except Exception as e:
                logger.error(f"Error deleting keys for {symbol}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {"connected": True, "key_prefix": self.key_prefix, "symbols": []}

        try:
            pattern = f"{self.key_prefix}ts:*"
            for key in self.client.scan_iter(match=pattern, count=100):
                info = self.client.ts().info(key)
                stats["symbols"].append(
                    {
                        "key": key,
                        "total_samples": info.total_samples,
                        "retention_ms": info.retention_msecs,
                        "labels": info.labels,
                    }
                )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")

        return stats

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


class FeatureCache:
    """
    Higher-level cache for storing computed features.
    Provides typed methods for common feature operations.
    """

    def __init__(self, cache: RedisTimeSeriesCache):
        self.cache = cache

    def store_gaf_image(self, symbol: str, image_type: str, image_data: bytes) -> bool:
        """Store GAF-encoded image as bytes"""
        key = f"{self.cache.key_prefix}gaf:{symbol}:{image_type}"

        try:
            self.cache.client.setex(key, 3600, image_data)  # 1 hour TTL
            return True
        except Exception as e:
            logger.error(f"Error storing GAF image: {e}")
            return False

    def get_gaf_image(self, symbol: str, image_type: str) -> Optional[bytes]:
        """Retrieve GAF-encoded image"""
        key = f"{self.cache.key_prefix}gaf:{symbol}:{image_type}"

        try:
            return self.cache.client.get(key)
        except Exception as e:
            logger.error(f"Error retrieving GAF image: {e}")
        return None

    def store_correlation_rings(self, symbol: str, rings_data: Dict) -> bool:
        """Store hierarchical correlation ring data"""
        key = f"{self.cache.key_prefix}rings:{symbol}"

        try:
            self.cache.client.setex(key, 3600, json.dumps(rings_data, default=str))
            return True
        except Exception as e:
            logger.error(f"Error storing correlation rings: {e}")
            return False

    def get_correlation_rings(self, symbol: str) -> Optional[Dict]:
        """Retrieve hierarchical correlation ring data"""
        key = f"{self.cache.key_prefix}rings:{symbol}"

        try:
            data = self.cache.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving correlation rings: {e}")
        return None

    def store_signal(self, symbol: str, signal_data: Dict) -> bool:
        """Store latest signal output"""
        key = f"{self.cache.key_prefix}signal:{symbol}"

        try:
            self.cache.client.setex(
                key,
                300,  # 5 minute TTL
                json.dumps(signal_data, default=str),
            )
            return True
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
            return False

    def get_latest_signal(self, symbol: str) -> Optional[Dict]:
        """Retrieve latest signal output"""
        key = f"{self.cache.key_prefix}signal:{symbol}"

        try:
            data = self.cache.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error retrieving signal: {e}")
        return None


if __name__ == "__main__":
    cache = RedisTimeSeriesCache()

    print("Creating TimeSeries for MNQ...")
    cache.create_time_series("MNQ", "15m")
    cache.create_time_series("MNQ", "1h")
    cache.create_time_series("ES", "15m")

    stats = cache.get_stats()
    print(f"Cache stats: {stats}")

    cache.close()
