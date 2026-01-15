"""
Alpha Vantage News Sentiment Integration

Fetches market news and sentiment data to inform trading signals.
"""

import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsSentiment:
    """News sentiment data container"""

    timestamp: str
    title: str
    source: str
    url: str
    sentiment_score: float
    sentiment_label: str
    ticker_sentiment: List[Dict[str, Any]]
    topic: str
    market_relevance_score: float


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment metrics for a time period"""

    avg_sentiment: float
    sentiment_std: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_relevance: float
    top_topics: List[str]
    market_sentiment_score: float
    sentiment_trend: float
    topic_sentiments: Dict[str, float]


class AlphaVantageNewsClient:
    """
    Client for Alpha Vantage News Sentiment API.

    API Documentation: https://www.alphavantage.co/documentation/#newsapi
    Free tier: 5 requests/minute, 500/day
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, rate_limit: float = 12.0):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def _rate_limit_wait(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        wait_time = 60.0 / self.rate_limit - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def fetch_news_sentiment(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        limit: int = 100,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
    ) -> List[NewsSentiment]:
        """
        Fetch news sentiment data (synchronous version).
        """
        self._rate_limit_wait()

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": min(limit, 1000),
        }

        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        cache_key = str(sorted(params.items()))
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return data

        try:
            import requests

            response = requests.get(self.BASE_URL, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                news_items = self._parse_news_sentiment(data)

                self._cache[cache_key] = (time.time(), news_items)
                return news_items
            else:
                logger.error(f"API error: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    async def fetch_news_sentiment_async(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        limit: int = 100,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
    ) -> List[NewsSentiment]:
        """
        Fetch news sentiment data (async version).
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.BASE_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "apikey": self.api_key,
                    "limit": min(limit, 1000),
                    **({"tickers": ",".join(tickers)} if tickers else {}),
                    **({"topics": ",".join(topics)} if topics else {}),
                    **({"time_from": time_from} if time_from else {}),
                    **({"time_to": time_to} if time_to else {}),
                },
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_news_sentiment(data)
                else:
                    logger.error(f"API error: {response.status}")
                    return []

    def _parse_news_sentiment(self, data: Dict) -> List[NewsSentiment]:
        """Parse API response into NewsSentiment objects"""
        news_items = []

        for item in data.get("feed", []):
            ticker_sentiment = []
            for ts in item.get("ticker_sentiment", []):
                ticker_sentiment.append(
                    {
                        "ticker": ts.get("ticker", ""),
                        "relevance_score": float(ts.get("relevance_score", 0)),
                        "sentiment_score": float(ts.get("sentiment_score", 0)),
                        "sentiment_label": ts.get("sentiment_label", "Neutral"),
                    }
                )

            topics = item.get("topics", [])
            topic = topics[0].get("topic", "general") if topics else "general"

            news_items.append(
                NewsSentiment(
                    timestamp=item.get("time_published", ""),
                    title=item.get("title", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    sentiment_score=float(item.get("overall_sentiment_score", 0)),
                    sentiment_label=item.get("overall_sentiment_label", "Neutral"),
                    ticker_sentiment=ticker_sentiment,
                    topic=topic,
                    market_relevance_score=float(
                        item.get("market_relevance_score", 0.5)
                    ),
                )
            )

        return news_items

    def fetch_market_sentiment(
        self,
        symbols: List[str] = None,
        hours_back: int = 24,
    ) -> AggregatedSentiment:
        """
        Fetch and aggregate market sentiment for a time period.
        """
        time_from = (datetime.utcnow() - timedelta(hours=hours_back)).strftime(
            "%Y%m%dT%H%M"
        )

        tickers = (
            [f"FOREX:{s}" if ":" not in s else s for s in symbols] if symbols else None
        )

        news = self.fetch_news_sentiment(
            tickers=tickers,
            limit=500,
            time_from=time_from,
        )

        if not news:
            return AggregatedSentiment(
                avg_sentiment=0,
                sentiment_std=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                avg_relevance=0,
                top_topics=[],
                market_sentiment_score=0,
                sentiment_trend=0,
                topic_sentiments={},
            )

        scores = [n.sentiment_score for n in news]
        avg_sentiment = np.mean(scores)
        sentiment_std = np.std(scores)

        bullish = sum(1 for n in news if n.sentiment_label == "Bullish")
        bearish = sum(1 for n in news if n.sentiment_label == "Bearish")
        neutral = len(news) - bullish - bearish

        avg_relevance = np.mean([n.market_relevance_score for n in news])

        topic_counts = {}
        topic_sentiments = {}
        for n in news:
            topic_counts[n.topic] = topic_counts.get(n.topic, 0) + 1
            if n.topic not in topic_sentiments:
                topic_sentiments[n.topic] = []
            topic_sentiments[n.topic].append(n.sentiment_score)

        top_topics = sorted(topic_counts.keys(), key=lambda x: -topic_counts[x])[:5]
        topic_avg_sentiments = {t: np.mean(s) for t, s in topic_sentiments.items()}

        total_positive = sum(max(0, s) for s in scores)
        total_negative = sum(abs(min(0, s)) for s in scores)
        sentiment_trend = (total_positive - total_negative) / (
            total_positive + total_negative + 0.001
        )

        market_sentiment_score = (
            (avg_sentiment + 1) / 2 * 0.5
            + (bullish / max(len(news), 1)) * 0.2
            + (avg_relevance) * 0.2
            + sentiment_trend * 0.1
        )

        return AggregatedSentiment(
            avg_sentiment=avg_sentiment,
            sentiment_std=sentiment_std,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            avg_relevance=avg_relevance,
            top_topics=top_topics,
            market_sentiment_score=market_sentiment_score,
            sentiment_trend=sentiment_trend,
            topic_sentiments=topic_avg_sentiments,
        )


def create_sentiment_features(
    current: AggregatedSentiment,
    window: List[AggregatedSentiment] = None,
) -> np.ndarray:
    """
    Convert aggregated sentiment to feature vector for model.
    """
    features = [
        current.avg_sentiment,
        current.sentiment_std,
        current.bullish_count / max(current.bullish_count + current.bearish_count, 1),
        current.avg_relevance,
        current.market_sentiment_score,
        current.sentiment_trend,
    ]

    if window and len(window) > 1:
        window_scores = [s.avg_sentiment for s in window]
        features.extend(
            [
                np.mean(window_scores),
                np.std(window_scores),
                window[-1].avg_sentiment - window[0].avg_sentiment,
            ]
        )
    else:
        features.extend([0, 0, 0])

    topic_strength = {
        "technology": 0,
        "financial_markets": 0,
        "economy_macro": 0,
        "energy": 0,
        "healthcare": 0,
        "consumer_cyclical": 0,
        "industrials": 0,
    }
    for topic in current.top_topics:
        if topic in topic_strength:
            topic_strength[topic] = current.topic_sentiments.get(topic, 0)

    features.extend(list(topic_strength.values()))
    features.extend([0] * (25 - len(features)))

    return np.array(features[:25], dtype=np.float32)


def get_sentiment_signal(
    sentiment: AggregatedSentiment,
    thresholds: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Generate trading signal from sentiment data.
    """
    if thresholds is None:
        thresholds = {
            "bullish_threshold": 0.1,
            "bearish_threshold": -0.1,
            "strong_bullish": 0.3,
            "strong_bearish": -0.3,
            "min_relevance": 0.3,
        }

    if sentiment.avg_relevance < thresholds["min_relevance"]:
        return {
            "signal": "NEUTRAL",
            "strength": 0,
            "reason": f"Low relevance ({sentiment.avg_relevance:.2f})",
        }

    if sentiment.avg_sentiment >= thresholds["strong_bullish"]:
        return {
            "signal": "STRONG_BULLISH",
            "strength": sentiment.avg_sentiment,
            "reason": f"Strong bullish sentiment ({sentiment.avg_sentiment:.2f})",
        }
    elif sentiment.avg_sentiment >= thresholds["bullish_threshold"]:
        return {
            "signal": "BULLISH",
            "strength": sentiment.avg_sentiment,
            "reason": f"Bullish sentiment ({sentiment.avg_sentiment:.2f})",
        }
    elif sentiment.avg_sentiment <= thresholds["strong_bearish"]:
        return {
            "signal": "STRONG_BEARISH",
            "strength": sentiment.avg_sentiment,
            "reason": f"Strong bearish sentiment ({sentiment.avg_sentiment:.2f})",
        }
    elif sentiment.avg_sentiment <= thresholds["bearish_threshold"]:
        return {
            "signal": "BEARISH",
            "strength": sentiment.avg_sentiment,
            "reason": f"Bearish sentiment ({sentiment.avg_sentiment:.2f})",
        }
    else:
        return {
            "signal": "NEUTRAL",
            "strength": 0,
            "reason": f"Neutral sentiment ({sentiment.avg_sentiment:.2f})",
        }


if __name__ == "__main__":
    import os

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    client = AlphaVantageNewsClient(api_key=api_key)

    print("Testing Alpha Vantage News Sentiment API...")
    sentiment = client.fetch_market_sentiment(
        symbols=["MNQ", "ES", "NQ", "QQQ"],
        hours_back=24,
    )

    print(f"\n=== Market Sentiment Report ===")
    print(f"Average Sentiment: {sentiment.avg_sentiment:.3f}")
    print(f"Sentiment Std: {sentiment.sentiment_std:.3f}")
    print(f"Market Score: {sentiment.market_sentiment_score:.3f}")
    print(f"Sentiment Trend: {sentiment.sentiment_trend:.3f}")
    print(
        f"Bullish: {sentiment.bullish_count}, Bearish: {sentiment.bearish_count}, Neutral: {sentiment.neutral_count}"
    )
    print(f"Top Topics: {sentiment.top_topics}")
    print(f"Topic Sentiments: {sentiment.topic_sentiments}")

    signal = get_sentiment_signal(sentiment)
    print(f"\n=== Sentiment Signal ===")
    print(f"Signal: {signal['signal']}")
    print(f"Strength: {signal['strength']:.3f}")
    print(f"Reason: {signal['reason']}")

    features = create_sentiment_features(sentiment)
    print(f"\n=== Feature Vector ===")
    print(f"Shape: {features.shape}")
    print(f"Features: {features[:10]}...")
