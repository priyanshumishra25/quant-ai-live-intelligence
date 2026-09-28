"""
Database schema: SQLAlchemy ORM models (PostgreSQL / TimescaleDB).
Run `alembic upgrade head` to apply migrations.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float,
    ForeignKey, Index, Integer, JSON, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Stocks universe
# ---------------------------------------------------------------------------

class Instrument(Base):
    __tablename__ = "instruments"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(16),  nullable=False, unique=True)
    name        = Column(String(256), nullable=True)
    exchange    = Column(String(32),  nullable=True)
    sector      = Column(String(64),  nullable=True)
    industry    = Column(String(128), nullable=True)
    asset_type  = Column(String(32),  nullable=False, default="equity")   # equity|etf|crypto|forex
    is_active   = Column(Boolean, nullable=False, default=True)
    created_at  = Column(DateTime, nullable=False, default=datetime.utcnow)

    prices   = relationship("OHLCVBar",     back_populates="instrument", cascade="all, delete-orphan")
    features = relationship("FeatureStore", back_populates="instrument", cascade="all, delete-orphan")
    signals  = relationship("Signal",       back_populates="instrument", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_instruments_ticker", "ticker"),
    )


# ---------------------------------------------------------------------------
# Market data (TimescaleDB hypertable for the `timestamp` column)
# ---------------------------------------------------------------------------

class OHLCVBar(Base):
    """
    OHLCV bars at multiple resolutions.
    After migration, convert to a TimescaleDB hypertable:
        SELECT create_hypertable('ohlcv_bars', 'timestamp');
    """
    __tablename__ = "ohlcv_bars"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    timestamp     = Column(DateTime, nullable=False)
    interval      = Column(String(8), nullable=False)   # 1m | 5m | 1h | 1d
    open          = Column(Float, nullable=False)
    high          = Column(Float, nullable=False)
    low           = Column(Float, nullable=False)
    close         = Column(Float, nullable=False)
    volume        = Column(BigInteger, nullable=True)
    adj_close     = Column(Float, nullable=True)
    vwap          = Column(Float, nullable=True)

    instrument = relationship("Instrument", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("instrument_id", "timestamp", "interval", name="uq_ohlcv"),
        Index("ix_ohlcv_ticker_ts", "instrument_id", "timestamp"),
    )


class MacroDataPoint(Base):
    __tablename__ = "macro_data"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    series    = Column(String(64), nullable=False)    # FEDFUNDS | CPI | ...
    timestamp = Column(DateTime,  nullable=False)
    value     = Column(Float,     nullable=False)

    __table_args__ = (
        UniqueConstraint("series", "timestamp", name="uq_macro"),
        Index("ix_macro_series_ts", "series", "timestamp"),
    )


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

class FeatureStore(Base):
    """
    Stores pre-computed daily feature vectors as JSON.
    Large deployments should move this to Redis or a dedicated feature store.
    """
    __tablename__ = "feature_store"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date          = Column(DateTime, nullable=False)
    feature_set   = Column(String(32), nullable=False, default="default")
    features      = Column(JSON, nullable=False)   # dict of feature_name → float
    created_at    = Column(DateTime, nullable=False, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="features")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", "feature_set", name="uq_features"),
        Index("ix_features_ticker_date", "instrument_id", "date"),
    )


# ---------------------------------------------------------------------------
# News & sentiment
# ---------------------------------------------------------------------------

class NewsArticle(Base):
    __tablename__ = "news_articles"

    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    source       = Column(String(64),   nullable=False)
    url          = Column(Text,         nullable=False, unique=True)
    title        = Column(Text,         nullable=False)
    body         = Column(Text,         nullable=True)
    published_at = Column(DateTime,     nullable=True)
    scraped_at   = Column(DateTime,     nullable=False, default=datetime.utcnow)
    tickers      = Column(JSON,         nullable=True)   # list[str]
    content_hash = Column(String(32),   nullable=False)

    sentiment = relationship("ArticleSentiment", back_populates="article", uselist=False)

    __table_args__ = (
        Index("ix_news_published", "published_at"),
    )


class ArticleSentiment(Base):
    __tablename__ = "article_sentiment"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id    = Column(BigInteger, ForeignKey("news_articles.id"), nullable=False, unique=True)
    sentiment     = Column(String(16), nullable=False)   # POSITIVE | NEGATIVE | NEUTRAL
    score         = Column(Float,      nullable=False)   # −1 to +1
    fear_score    = Column(Float,      nullable=True)
    greed_score   = Column(Float,      nullable=True)
    model_version = Column(String(32), nullable=True)
    scored_at     = Column(DateTime,   nullable=False, default=datetime.utcnow)

    article = relationship("NewsArticle", back_populates="sentiment")


class SocialPost(Base):
    __tablename__ = "social_posts"

    id         = Column(BigInteger, primary_key=True, autoincrement=True)
    platform   = Column(String(32),  nullable=False)   # reddit | twitter | stocktwits
    post_id    = Column(String(128), nullable=False)
    subreddit  = Column(String(64),  nullable=True)
    author     = Column(String(128), nullable=True)
    text       = Column(Text,        nullable=False)
    tickers    = Column(JSON,        nullable=True)
    upvotes    = Column(Integer,     nullable=True, default=0)
    comments   = Column(Integer,     nullable=True, default=0)
    created_at = Column(DateTime,    nullable=False)
    scraped_at = Column(DateTime,    nullable=False, default=datetime.utcnow)
    score      = Column(Float,       nullable=True)

    __table_args__ = (
        UniqueConstraint("platform", "post_id", name="uq_social_post"),
        Index("ix_social_created", "created_at"),
    )


class AggregatedSentiment(Base):
    """Daily/hourly aggregated sentiment per ticker."""
    __tablename__ = "aggregated_sentiment"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    timestamp     = Column(DateTime, nullable=False)
    interval      = Column(String(8), nullable=False, default="1d")
    bullish_score = Column(Float, nullable=True)
    bearish_score = Column(Float, nullable=True)
    fear_index    = Column(Float, nullable=True)
    greed_index   = Column(Float, nullable=True)
    article_count = Column(Integer, nullable=True)
    post_count    = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("instrument_id", "timestamp", "interval", name="uq_agg_sentiment"),
    )


# ---------------------------------------------------------------------------
# Predictions & signals
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id   = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    generated_at    = Column(DateTime, nullable=False, default=datetime.utcnow)
    target_date     = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=True)
    predicted_return = Column(Float, nullable=True)
    direction       = Column(String(16), nullable=True)   # UP | DOWN | SIDEWAYS
    confidence      = Column(Float, nullable=True)
    ci_lower        = Column(Float, nullable=True)
    ci_upper        = Column(Float, nullable=True)
    predicted_vol   = Column(Float, nullable=True)
    model_version   = Column(String(32), nullable=True)
    model_weights   = Column(JSON,  nullable=True)
    shap_values     = Column(JSON,  nullable=True)
    explanation     = Column(Text,  nullable=True)

    __table_args__ = (
        Index("ix_predictions_ticker_date", "instrument_id", "target_date"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    generated_at  = Column(DateTime, nullable=False, default=datetime.utcnow)
    signal        = Column(String(16), nullable=False)   # STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
    confidence    = Column(Float, nullable=True)
    risk_score    = Column(Float, nullable=True)
    entry_price   = Column(Float, nullable=True)
    target_price  = Column(Float, nullable=True)
    stop_loss     = Column(Float, nullable=True)
    horizon_days  = Column(Integer, nullable=True)

    instrument = relationship("Instrument", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_ticker_date", "instrument_id", "generated_at"),
    )


# ---------------------------------------------------------------------------
# Model registry / performance tracking
# ---------------------------------------------------------------------------

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(64), nullable=False)
    version       = Column(String(32), nullable=False)
    mlflow_run_id = Column(String(64), nullable=True)
    ticker        = Column(String(16), nullable=True)   # NULL = global model
    train_end     = Column(DateTime,   nullable=True)
    rmse          = Column(Float,      nullable=True)
    sharpe        = Column(Float,      nullable=True)
    dir_accuracy  = Column(Float,      nullable=True)
    is_active     = Column(Boolean,    nullable=False, default=True)
    created_at    = Column(DateTime,   nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_version"),
    )


class ModelPerformanceLog(Base):
    """Daily live-trading performance log per model version."""
    __tablename__ = "model_performance_logs"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    date             = Column(DateTime, nullable=False)
    predicted_return = Column(Float,    nullable=True)
    actual_return    = Column(Float,    nullable=True)
    direction_hit    = Column(Boolean,  nullable=True)
    abs_error        = Column(Float,    nullable=True)

    __table_args__ = (
        Index("ix_perf_log_model_date", "model_version_id", "date"),
    )


# ---------------------------------------------------------------------------
# Portfolio / trades (for paper and live trading)
# ---------------------------------------------------------------------------

class Portfolio(Base):
    __tablename__ = "portfolios"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    name         = Column(String(64), nullable=False, unique=True)
    description  = Column(Text, nullable=True)
    cash         = Column(Float, nullable=False, default=100_000.0)
    is_live      = Column(Boolean, nullable=False, default=False)
    created_at   = Column(DateTime, nullable=False, default=datetime.utcnow)

    trades = relationship("Trade", back_populates="portfolio")


class Trade(Base):
    __tablename__ = "trades"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    portfolio_id  = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    side          = Column(String(8), nullable=False)    # BUY | SELL
    quantity      = Column(Float,     nullable=False)
    entry_price   = Column(Float,     nullable=False)
    exit_price    = Column(Float,     nullable=True)
    entry_time    = Column(DateTime,  nullable=False)
    exit_time     = Column(DateTime,  nullable=True)
    commission    = Column(Float,     nullable=False, default=0.0)
    slippage      = Column(Float,     nullable=False, default=0.0)
    pnl           = Column(Float,     nullable=True)
    signal_id     = Column(BigInteger, ForeignKey("signals.id"), nullable=True)

    portfolio = relationship("Portfolio", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_portfolio_entry", "portfolio_id", "entry_time"),
    )
