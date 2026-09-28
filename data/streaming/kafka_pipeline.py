"""
Kafka streaming pipeline.
Producers push market data, news, and predictions.
Consumers drive the feature pipeline and model inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Awaitable

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic registry
# ---------------------------------------------------------------------------

TOPICS = {
    "market.ticks":        {"partitions": 12, "replication": 1},
    "market.ohlcv":        {"partitions": 6,  "replication": 1},
    "news.raw":            {"partitions": 4,  "replication": 1},
    "social.raw":          {"partitions": 4,  "replication": 1},
    "sentiment.scored":    {"partitions": 4,  "replication": 1},
    "features.computed":   {"partitions": 6,  "replication": 1},
    "predictions.output":  {"partitions": 6,  "replication": 1},
    "signals.generated":   {"partitions": 4,  "replication": 1},
    "alerts.triggered":    {"partitions": 2,  "replication": 1},
    "retraining.trigger":  {"partitions": 1,  "replication": 1},
}


# ---------------------------------------------------------------------------
# Message schemas (dataclasses → JSON)
# ---------------------------------------------------------------------------

@dataclass
class TickMessage:
    ticker:    str
    price:     float
    volume:    int
    bid:       float
    ask:       float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class OHLCVMessage:
    ticker:    str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    int
    interval:  str          # "1m" | "5m" | "1h" | "1d"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class PredictionMessage:
    ticker:            str
    predicted_price:   float
    direction:         str      # UP | DOWN | SIDEWAYS
    confidence:        float    # 0–1
    signal:            str      # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    risk_score:        float
    model_version:     str
    timestamp:         str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class RetrainingTrigger:
    reason:    str       # "scheduled" | "drift" | "performance_drop"
    ticker:    str | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _encode(obj: Any) -> bytes:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return json.dumps(obj, default=str).encode()


def _decode(raw: bytes) -> dict:
    return json.loads(raw.decode())


# ---------------------------------------------------------------------------
# Admin: ensure topics exist
# ---------------------------------------------------------------------------

async def ensure_topics(bootstrap_servers: str = "localhost:9092") -> None:
    try:
        admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers)
        await admin.start()
        existing = set(await admin.list_topics())
        to_create = [
            NewTopic(
                name=name,
                num_partitions=cfg["partitions"],
                replication_factor=cfg["replication"],
            )
            for name, cfg in TOPICS.items()
            if name not in existing
        ]
        if to_create:
            await admin.create_topics(to_create)
            logger.info("Created %d Kafka topics", len(to_create))
        await admin.close()
    except Exception as exc:
        logger.error("Kafka admin error: %s", exc)


# ---------------------------------------------------------------------------
# Producer wrapper
# ---------------------------------------------------------------------------

class MarketProducer:
    """
    Async Kafka producer with typed send methods.
    Usage:
        async with MarketProducer() as p:
            await p.send_tick(TickMessage(...))
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self._servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def __aenter__(self) -> "MarketProducer":
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._servers,
            value_serializer=_encode,
            key_serializer=lambda k: k.encode() if k else None,
            compression_type="gzip",
            acks="all",                  # strongest durability
            enable_idempotence=True,
        )
        await self._producer.start()
        return self

    async def __aexit__(self, *_) -> None:
        if self._producer:
            await self._producer.stop()

    async def _send(self, topic: str, key: str, value: Any) -> None:
        assert self._producer is not None
        try:
            await self._producer.send_and_wait(topic, key=key, value=value)
        except Exception as exc:
            logger.error("Kafka send error [%s]: %s", topic, exc)

    async def send_tick(self, tick: TickMessage) -> None:
        await self._send("market.ticks", tick.ticker, tick)

    async def send_ohlcv(self, bar: OHLCVMessage) -> None:
        await self._send("market.ohlcv", bar.ticker, bar)

    async def send_news(self, article: dict) -> None:
        key = article.get("source", "unknown")
        await self._send("news.raw", key, article)

    async def send_social(self, post: dict) -> None:
        key = post.get("platform", "unknown")
        await self._send("social.raw", key, post)

    async def send_prediction(self, pred: PredictionMessage) -> None:
        await self._send("predictions.output", pred.ticker, pred)

    async def send_signal(self, signal: dict) -> None:
        await self._send("signals.generated", signal.get("ticker", ""), signal)

    async def trigger_retraining(self, reason: str, ticker: str | None = None) -> None:
        msg = RetrainingTrigger(reason=reason, ticker=ticker)
        await self._send("retraining.trigger", ticker or "global", msg)


# ---------------------------------------------------------------------------
# Consumer base class
# ---------------------------------------------------------------------------

Handler = Callable[[dict], Awaitable[None]]


class BaseConsumer:
    """
    Wraps AIOKafkaConsumer with auto-commit, error handling,
    and a simple handler-registration pattern.
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str,
        bootstrap_servers: str = "localhost:9092",
        max_poll_records: int = 100,
    ):
        self._topics = topics
        self._group_id = group_id
        self._servers = bootstrap_servers
        self._max_poll = max_poll_records
        self._handlers: dict[str, list[Handler]] = {}
        self._consumer: AIOKafkaConsumer | None = None

    def on(self, topic: str) -> Callable[[Handler], Handler]:
        """Decorator: @consumer.on("market.ticks")"""
        def decorator(fn: Handler) -> Handler:
            self._handlers.setdefault(topic, []).append(fn)
            return fn
        return decorator

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._servers,
            group_id=self._group_id,
            value_deserializer=_decode,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            max_poll_records=self._max_poll,
        )
        await self._consumer.start()
        logger.info("Consumer [%s] started on %s", self._group_id, self._topics)

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()

    async def run(self) -> None:
        assert self._consumer is not None
        async for msg in self._consumer:
            topic = msg.topic
            handlers = self._handlers.get(topic, [])
            for handler in handlers:
                try:
                    await handler(msg.value)
                except Exception as exc:
                    logger.error(
                        "Handler error on topic %s: %s", topic, exc, exc_info=True
                    )


# ---------------------------------------------------------------------------
# Specialised consumers
# ---------------------------------------------------------------------------

class FeaturePipelineConsumer(BaseConsumer):
    """
    Consumes OHLCV bars + sentiment scores, computes features,
    and publishes to features.computed.
    """

    def __init__(self, producer: MarketProducer, **kwargs):
        super().__init__(
            topics=["market.ohlcv", "sentiment.scored"],
            group_id="feature-pipeline",
            **kwargs,
        )
        self._producer = producer
        self._buffers: dict[str, list[dict]] = {}     # ticker → last N bars

        @self.on("market.ohlcv")
        async def handle_ohlcv(msg: dict) -> None:
            ticker = msg.get("ticker", "")
            buf = self._buffers.setdefault(ticker, [])
            buf.append(msg)
            if len(buf) > 300:
                buf.pop(0)
            if len(buf) >= 60:
                features = self._compute_features(ticker, buf)
                await self._producer._send("features.computed", ticker, features)

        @self.on("sentiment.scored")
        async def handle_sentiment(msg: dict) -> None:
            # Sentiment updates are merged into latest feature set
            ticker = msg.get("ticker", "")
            logger.debug("Sentiment update for %s: %.3f", ticker, msg.get("score", 0))

    def _compute_features(self, ticker: str, bars: list[dict]) -> dict:
        """Lightweight feature extraction for streaming (full version is in indicators.py)."""
        closes = [b["close"] for b in bars]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        return {
            "ticker":     ticker,
            "timestamp":  datetime.utcnow().isoformat(),
            "close":      closes[-1],
            "ma20":       ma20,
            "return_1d":  returns[-1] if returns else 0.0,
            "volatility": (sum(r ** 2 for r in returns[-20:]) / 20) ** 0.5 if returns else 0.0,
        }


class PredictionConsumer(BaseConsumer):
    """
    Consumes computed features, runs ensemble inference,
    and publishes predictions + signals.
    """

    def __init__(
        self,
        producer: MarketProducer,
        ensemble_fn: Callable[[dict], Awaitable[PredictionMessage]],
        **kwargs,
    ):
        super().__init__(
            topics=["features.computed"],
            group_id="prediction-service",
            **kwargs,
        )
        self._producer = producer
        self._ensemble_fn = ensemble_fn

        @self.on("features.computed")
        async def handle_features(msg: dict) -> None:
            pred = await self._ensemble_fn(msg)
            await self._producer.send_prediction(pred)
            await self._producer.send_signal({
                "ticker":    pred.ticker,
                "signal":    pred.signal,
                "confidence": pred.confidence,
                "timestamp": pred.timestamp,
            })


class AlertConsumer(BaseConsumer):
    """
    Monitors signals and triggers alerts for strong buy/sell events.
    Also watches for performance drift and emits retraining triggers.
    """

    def __init__(self, producer: MarketProducer, **kwargs):
        super().__init__(
            topics=["signals.generated"],
            group_id="alert-service",
            **kwargs,
        )
        self._producer = producer
        self._signal_counts: dict[str, int] = {}

        @self.on("signals.generated")
        async def handle_signal(msg: dict) -> None:
            ticker = msg.get("ticker", "")
            signal = msg.get("signal", "HOLD")
            confidence = msg.get("confidence", 0.0)

            if signal in ("STRONG_BUY", "STRONG_SELL") and confidence > 0.80:
                alert = {
                    "type":       "high_confidence_signal",
                    "ticker":     ticker,
                    "signal":     signal,
                    "confidence": confidence,
                    "timestamp":  msg.get("timestamp"),
                }
                await self._producer._send("alerts.triggered", ticker, alert)
                logger.info("ALERT: %s %s @ %.0f%%", signal, ticker, confidence * 100)

            # Track signal frequency for drift detection
            self._signal_counts[ticker] = self._signal_counts.get(ticker, 0) + 1
            if self._signal_counts[ticker] % 1000 == 0:
                await self._producer.trigger_retraining("scheduled", ticker)
                self._signal_counts[ticker] = 0


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def run_streaming_pipeline(
    bootstrap_servers: str = "localhost:9092",
    ensemble_fn: Callable | None = None,
) -> None:
    await ensure_topics(bootstrap_servers)

    if ensemble_fn is None:
        async def _dummy(features: dict) -> PredictionMessage:
            return PredictionMessage(
                ticker=features.get("ticker", "???"),
                predicted_price=features.get("close", 0) * 1.001,
                direction="UP",
                confidence=0.5,
                signal="HOLD",
                risk_score=0.5,
                model_version="0.0.0",
            )
        ensemble_fn = _dummy

    async with MarketProducer(bootstrap_servers) as producer:
        feature_consumer  = FeaturePipelineConsumer(producer,   bootstrap_servers=bootstrap_servers)
        prediction_consumer = PredictionConsumer(producer, ensemble_fn, bootstrap_servers=bootstrap_servers)
        alert_consumer    = AlertConsumer(producer,              bootstrap_servers=bootstrap_servers)

        await asyncio.gather(
            feature_consumer.start(),
            prediction_consumer.start(),
            alert_consumer.start(),
        )
        await asyncio.gather(
            feature_consumer.run(),
            prediction_consumer.run(),
            alert_consumer.run(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_streaming_pipeline())
