"""Central configuration for the Quant AI research platform."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class DataSourceSettings(BaseSettings):
    market_data_provider: Literal["bundled", "yfinance"] = Field(
        default="bundled", validation_alias="MARKET_DATA_PROVIDER"
    )
    sample_data_dir: Path = Field(
        default=ROOT / "data" / "sample", validation_alias="SAMPLE_DATA_DIR"
    )
    alpha_vantage_key: str = Field(default="", validation_alias="ALPHA_VANTAGE_KEY")
    alpha_vantage_outputsize: Literal["compact", "full"] = Field(
        default="compact", validation_alias="ALPHA_VANTAGE_OUTPUTSIZE"
    )
    alpha_vantage_min_interval_seconds: float = Field(
        default=1.10, validation_alias="ALPHA_VANTAGE_MIN_INTERVAL_SECONDS"
    )
    polygon_key: str = Field(default="", validation_alias="POLYGON_KEY")
    newsapi_key: str = Field(default="", validation_alias="NEWSAPI_KEY")
    reddit_client_id: str = Field(default="", validation_alias="REDDIT_CLIENT_ID")
    reddit_secret: str = Field(default="", validation_alias="REDDIT_SECRET")
    reddit_user_agent: str = Field(default="QuantAIResearch/3.0", validation_alias="REDDIT_USER_AGENT")
    intelligence_lookback_days: int = Field(default=365, validation_alias="INTELLIGENCE_LOOKBACK_DAYS")
    intelligence_news_limit: int = Field(default=500, validation_alias="INTELLIGENCE_NEWS_LIMIT")
    intelligence_reddit_limit: int = Field(default=100, validation_alias="INTELLIGENCE_REDDIT_LIMIT")
    intelligence_reddit_comment_posts: int = Field(default=5, validation_alias="INTELLIGENCE_REDDIT_COMMENT_POSTS")
    intelligence_reddit_comments_per_post: int = Field(default=20, validation_alias="INTELLIGENCE_REDDIT_COMMENTS_PER_POST")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ModelSettings(BaseSettings):
    artifact_dir: Path = Field(
        default=ROOT / "artifacts" / "models", validation_alias="MODEL_ARTIFACT_DIR"
    )
    default_horizon: Literal["1d", "5d", "20d"] = Field(
        default="5d", validation_alias="DEFAULT_HORIZON"
    )
    seed: int = Field(default=42, validation_alias="MODEL_SEED")

    # Retained settings for the optional experimental research modules.
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.3
    lstm_seq_len: int = 60
    lstm_pred_len: int = 5
    tft_hidden_size: int = 128
    tft_attention_heads: int = 4
    tft_lstm_layers: int = 2
    tft_dropout: float = 0.1
    tft_max_encoder_length: int = 60
    tft_max_prediction_length: int = 5
    xgb_n_estimators: int = 1000
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.01
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_early_stopping_rounds: int = 50
    lgb_n_estimators: int = 1000
    lgb_num_leaves: int = 63
    lgb_learning_rate: float = 0.01
    ensemble_method: Literal["stacking", "blending", "dynamic_weighting"] = "dynamic_weighting"
    ensemble_lookback_days: int = 30
    rl_algorithm: Literal["PPO", "SAC", "TD3", "A2C"] = "PPO"
    rl_total_timesteps: int = 1_000_000
    rl_n_envs: int = 8
    rl_learning_rate: float = 3e-4
    batch_size: int = 64
    val_batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    wf_initial_train_size: int = 504
    wf_step_size: int = 63
    wf_n_splits: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class InfraSettings(BaseSettings):
    cache_backend: Literal["memory", "redis", "auto"] = Field(
        default="memory", validation_alias="CACHE_BACKEND"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    redis_ttl_predictions: int = Field(default=300, validation_alias="REDIS_TTL_PREDICTIONS")
    redis_ttl_intelligence: int = Field(default=1800, validation_alias="REDIS_TTL_INTELLIGENCE")
    redis_ttl_symbol_resolution: int = Field(default=604800, validation_alias="REDIS_TTL_SYMBOL_RESOLUTION")
    redis_ttl_market_history: int = Field(default=21600, validation_alias="REDIS_TTL_MARKET_HISTORY")
    redis_ttl_news: int = Field(default=1800, validation_alias="REDIS_TTL_NEWS")
    redis_ttl_reddit: int = Field(default=600, validation_alias="REDIS_TTL_REDDIT")
    postgres_url: str = Field(
        default="postgresql+asyncpg://quantai:quantai@localhost:5432/quantai",
        validation_alias="POSTGRES_URL",
    )
    mongo_url: str = Field(default="mongodb://localhost:27017", validation_alias="MONGO_URL")
    kafka_brokers: str = Field(default="localhost:9092", validation_alias="KAFKA_BROKERS")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = "quant-ai-models"
    s3_bucket: str = Field(default="quant-ai-models", validation_alias="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", validation_alias="S3_REGION")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class APISettings(BaseSettings):
    host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    port: int = Field(default=8000, validation_alias="API_PORT")
    workers: int = Field(default=1, validation_alias="API_WORKERS")
    jwt_secret: str = Field(
        default="development-only-change-me-please-32-bytes",
        validation_alias="JWT_SECRET",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=60, validation_alias="JWT_EXPIRE_MINUTES")
    auth_username: str = Field(default="demo", validation_alias="AUTH_USERNAME")
    auth_password: str = Field(default="quantai-demo", validation_alias="AUTH_PASSWORD")
    auth_required: bool = Field(default=True, validation_alias="AUTH_REQUIRED")
    rate_limit_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_PER_MINUTE")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:8080"],
        validation_alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class TradingSettings(BaseSettings):
    equities: list[str] = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.20
    max_drawdown_pct: float = 0.15
    var_confidence: float = 0.95
    kelly_fraction: float = 0.25
    commission_pct: float = 0.0010
    slippage_pct: float = 0.0005
    short_borrow_rate: float = 0.02
    price_poll_interval: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    env: Literal["development", "test", "staging", "production"] = Field(
        default="development", validation_alias="ENV"
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_json: bool = Field(default=False, validation_alias="LOG_JSON")
    data: DataSourceSettings = DataSourceSettings()
    model: ModelSettings = ModelSettings()
    infra: InfraSettings = InfraSettings()
    api: APISettings = APISettings()
    trading: TradingSettings = TradingSettings()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
