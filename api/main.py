"""Runnable FastAPI surface for the Quant AI research platform.

Default behaviour is deliberately offline and reproducible: predictions use
persisted lightweight model artefacts over bundled deterministic sample data.
Switching to a live data provider is explicit and does not change the model's
research/non-advisory status.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Any

import jwt
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from backtesting.engine import BacktestEngine
from config import get_settings
from services.auth import authenticate, create_access_token, decode_access_token
from services.cache import create_cache, get_json, set_json
from services.market_data import MarketDataError, MarketDataService
from services.prediction_service import ArtifactRegistry, HORIZON_DAYS, PredictionService, WalkForwardLinearSignal
from services.experiments import MODEL_CATALOG, build_signal
from services.live_intelligence import (
    AlphaVantageClient, LiveIntelligenceError, LiveIntelligenceService, ProviderLimitError, RedditClient,
)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    HTTP_REQUESTS = Counter("quant_ai_http_requests_total", "HTTP requests", ["method", "path", "status"])
    HTTP_SECONDS = Histogram("quant_ai_http_request_seconds", "HTTP request duration", ["method", "path"])
except ImportError:  # metrics remain optional for minimal source usage
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    HTTP_REQUESTS = HTTP_SECONDS = None
    generate_latest = None

settings = get_settings()
api_cfg = settings.api
security = HTTPBearer(auto_error=False)


class FixedWindowRateLimiter:
    """Per-process limiter suitable for the reference deployment.

    For a multi-replica public service, enforce rate limits at the gateway or
    replace this with a shared Redis-backed limiter.
    """
    def __init__(self, per_minute: int):
        self.limit = max(1, per_minute)
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.hits[key]
        while q and q[0] <= now - 60.0:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


rate_limiter = FixedWindowRateLimiter(api_cfg.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    cache, cache_backend = await create_cache(settings.infra.cache_backend, settings.infra.redis_url)
    market = MarketDataService(settings.data.market_data_provider, settings.data.sample_data_dir)
    registry = ArtifactRegistry(settings.model.artifact_dir)
    predictor = PredictionService(market, registry)
    intelligence = LiveIntelligenceService(
        AlphaVantageClient(
            settings.data.alpha_vantage_key,
            outputsize=settings.data.alpha_vantage_outputsize,
            min_interval_seconds=settings.data.alpha_vantage_min_interval_seconds,
        ),
        RedditClient(
            settings.data.reddit_client_id,
            settings.data.reddit_secret,
            settings.data.reddit_user_agent,
        ),
        lookback_days=settings.data.intelligence_lookback_days,
        news_fetch_limit=settings.data.intelligence_news_limit,
        reddit_fetch_limit=settings.data.intelligence_reddit_limit,
        reddit_comment_posts=settings.data.intelligence_reddit_comment_posts,
        reddit_comments_per_post=settings.data.intelligence_reddit_comments_per_post,
        cache=cache,
        symbol_cache_ttl=settings.infra.redis_ttl_symbol_resolution,
        history_cache_ttl=settings.infra.redis_ttl_market_history,
        news_cache_ttl=settings.infra.redis_ttl_news,
        reddit_cache_ttl=settings.infra.redis_ttl_reddit,
    )
    app.state.cache = cache
    app.state.cache_backend = cache_backend
    app.state.market = market
    app.state.predictor = predictor
    app.state.intelligence = intelligence
    app.state.runtime = {
        "started_at": time.time(),
        "requests_total": 0,
        "http_seconds_sum": 0.0,
        "prediction_requests": 0,
        "prediction_cache_hits": 0,
    }
    yield
    await intelligence.close()
    await cache.close()


app = FastAPI(
    title="Quant AI Live Intelligence Platform",
    description=(
        "Software + AI engineering portfolio API with a reproducible offline path and an optional live "
        "intelligence pipeline that fuses historical trends, documented news feeds and Reddit evidence."
    ),
    version="3.0.1",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def apply_rate_limit(request: Request, call_next):
    started = time.perf_counter()
    if request.url.path.startswith("/api/v1/") and request.url.path not in {"/api/v1/auth/token"}:
        host = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(host):
            from fastapi.responses import JSONResponse
            response = JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
            if HTTP_REQUESTS is not None:
                HTTP_REQUESTS.labels(request.method, request.url.path, "429").inc()
            return response
    response = await call_next(request)
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        runtime["requests_total"] += 1
        runtime["http_seconds_sum"] += time.perf_counter() - started
    if HTTP_REQUESTS is not None and HTTP_SECONDS is not None:
        HTTP_REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
        HTTP_SECONDS.labels(request.method, request.url.path).observe(time.perf_counter() - started)
    return response


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class PredictionRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=16, examples=["AAPL"])
    horizon: str = Field(default="5d", pattern="^(1d|5d|20d)$")
    include_sentiment: bool = True
    include_explainability: bool = True


class BatchPredictionRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=50)
    horizon: str = Field(default="5d", pattern="^(1d|5d|20d)$")


class BacktestRequest(BaseModel):
    ticker: str = "SPY"
    horizon: str = Field(default="5d", pattern="^(1d|5d|20d)$")
    initial_equity: float = Field(default=100_000.0, gt=1_000.0)
    train_size: int = Field(default=300, ge=120, le=700)
    step_size: int = Field(default=63, ge=20, le=252)
    n_splits: int = Field(default=6, ge=1, le=12)


class ExperimentRequest(BaseModel):
    ticker: str = "SPY"
    horizon: str = Field(default="5d", pattern="^(1d|5d|20d)$")
    models: list[str] = Field(default_factory=lambda: ["ridge", "momentum", "mean_reversion"], min_length=1, max_length=3)
    initial_equity: float = Field(default=100_000.0, gt=1_000.0)
    train_size: int = Field(default=300, ge=120, le=700)
    step_size: int = Field(default=63, ge=20, le=252)
    n_splits: int = Field(default=4, ge=1, le=12)
    commission_bps: float = Field(default=10.0, ge=0.0, le=100.0)
    slippage_bps: float = Field(default=5.0, ge=0.0, le=100.0)


class LiveIntelligenceRequest(BaseModel):
    query: str = Field(min_length=1, max_length=120, examples=["Apple", "AAPL", "Tesla"])
    horizon: str = Field(default="5d", pattern="^(1d|5d|20d)$")
    include_reddit_comments: bool = True


async def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, Any]:
    if not api_cfg.auth_required:
        return {"sub": "anonymous"}
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return decode_access_token(credentials.credentials, api_cfg.jwt_secret, api_cfg.jwt_algorithm)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None


def _services(request: Request) -> tuple[MarketDataService, PredictionService]:
    return request.app.state.market, request.app.state.predictor


@app.get("/", include_in_schema=False)
async def root():
    return {"name": "Quant AI Live Intelligence Platform", "version": "3.0.1", "docs": "/api/docs"}


@app.get("/metrics", include_in_schema=False)
async def metrics():
    if generate_latest is None:
        return Response("# prometheus-client is not installed\n", media_type="text/plain")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
async def health(request: Request):
    cache_ok = False
    try:
        cache_ok = bool(await request.app.state.cache.ping())
    except Exception:
        cache_ok = False
    predictor: PredictionService = request.app.state.predictor
    return {
        "status": "ok" if predictor.loaded else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.1",
        "models_loaded": predictor.loaded,
        "model_count": len(predictor.registry.available()),
        "cache_ok": cache_ok,
        "cache_backend": request.app.state.cache_backend,
        "data_provider": predictor.market.provider,
        "live_intelligence_configured": bool(request.app.state.intelligence.configured),
        "reddit_configured": bool(request.app.state.intelligence.reddit.configured),
        "data_mode": "bundled_synthetic_fixture" if predictor.market.provider == "bundled" else "external_market_data",
    }


@app.post("/api/v1/auth/token", response_model=TokenResponse, tags=["auth"])
@app.post("/auth/token", response_model=TokenResponse, include_in_schema=False)
async def token(body: TokenRequest):
    if not authenticate(body.username, body.password, api_cfg.auth_username, api_cfg.auth_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    encoded, expires = create_access_token(
        body.username, api_cfg.jwt_secret, api_cfg.jwt_algorithm, api_cfg.jwt_expire_minutes
    )
    return TokenResponse(access_token=encoded, expires_in=expires)


@app.get("/api/v1/models", tags=["models"])
async def models(request: Request, user: dict = Depends(current_user)):
    predictor: PredictionService = request.app.state.predictor
    return {
        "artifacts": predictor.registry.available(),
        "tickers": predictor.market.available_tickers(),
        "horizons": list(HORIZON_DAYS),
        "data_provider": predictor.market.provider,
        "live_intelligence_configured": bool(request.app.state.intelligence.configured),
        "reddit_configured": bool(request.app.state.intelligence.reddit.configured),
    }


async def _predict(request: Request, ticker: str, horizon: str) -> dict[str, Any]:
    cache_key = f"pred:v3:{ticker.upper()}:{horizon}:{settings.data.market_data_provider}"
    cached = await get_json(request.app.state.cache, cache_key)
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is not None:
        runtime["prediction_requests"] += 1
    if cached:
        cached["cache_hit"] = True
        if runtime is not None:
            runtime["prediction_cache_hits"] += 1
        return cached
    predictor: PredictionService = request.app.state.predictor
    started = time.perf_counter()
    try:
        result = predictor.predict(ticker, horizon)
    except (MarketDataError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["cache_hit"] = False
    await set_json(request.app.state.cache, cache_key, result, settings.infra.redis_ttl_predictions)
    return result


@app.post("/api/v1/predict", tags=["predictions"])
async def predict(body: PredictionRequest, request: Request, user: dict = Depends(current_user)):
    return await _predict(request, body.ticker, body.horizon)


@app.get("/api/v1/predict/{ticker}", tags=["predictions"])
async def predict_get(ticker: str, request: Request, horizon: str = "5d", user: dict = Depends(current_user)):
    if horizon not in HORIZON_DAYS:
        raise HTTPException(status_code=422, detail="horizon must be one of 1d, 5d, 20d")
    return await _predict(request, ticker, horizon)


@app.post("/api/v1/predict/batch", tags=["predictions"])
async def predict_batch(body: BatchPredictionRequest, request: Request, user: dict = Depends(current_user)):
    unique = list(dict.fromkeys(t.upper() for t in body.tickers))[:50]
    results = []
    for ticker in unique:
        results.append(await _predict(request, ticker, body.horizon))
    return {"predictions": results, "count": len(results), "horizon": body.horizon}


@app.get("/api/v1/market/history/{ticker}", tags=["market"])
async def market_history(ticker: str, request: Request, limit: int = 120, user: dict = Depends(current_user)):
    market: MarketDataService = request.app.state.market
    try:
        df = market.history(ticker).tail(max(5, min(limit, 500)))
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "ticker": ticker.upper(),
        "data_mode": "bundled_synthetic_fixture" if market.provider == "bundled" else "external_market_data",
        "rows": [
            {"date": idx.date().isoformat(), **{k: float(row[k]) for k in ["open", "high", "low", "close"]}, "volume": int(row["volume"])}
            for idx, row in df.iterrows()
        ],
    }


@app.get("/api/v1/market/overview", tags=["market"])
async def market_overview(request: Request, user: dict = Depends(current_user)):
    market: MarketDataService = request.app.state.market
    tickers = market.available_tickers() or settings.trading.equities
    movers = []
    changes_20 = []
    sector_returns: dict[str, list[float]] = {"Technology": [], "Broad ETF": []}
    for ticker in tickers:
        try:
            df = market.history(ticker)
        except MarketDataError:
            continue
        ret1 = float(df["close"].pct_change().iloc[-1] * 100)
        ret20 = float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100) if len(df) > 21 else 0.0
        movers.append({"ticker": ticker, "return_pct": round(ret1, 3)})
        changes_20.append(ret20)
        sector_returns["Broad ETF" if ticker in {"SPY", "QQQ"} else "Technology"].append(ret1)
    movers.sort(key=lambda x: abs(x["return_pct"]), reverse=True)
    try:
        spy = market.history("SPY")
        vix_proxy = float(spy["close"].pct_change().tail(20).std() * np.sqrt(252) * 100)
    except Exception:
        vix_proxy = 0.0
    breadth = np.mean([1.0 if r > 0 else 0.0 for r in changes_20]) if changes_20 else 0.5
    momentum = float(np.tanh(np.mean(changes_20) / 8.0)) if changes_20 else 0.0
    fear_greed = float(np.clip(50 + 30 * (breadth - 0.5) * 2 + 20 * momentum, 0, 100))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_mode": "bundled_synthetic_fixture" if market.provider == "bundled" else "external_market_data",
        "fear_greed_index": round(fear_greed, 2),
        "vix_proxy": round(vix_proxy, 2),
        "top_movers": movers[:5],
        "sector_performance": {
            name: round(float(np.mean(values)), 3) if values else 0.0
            for name, values in sector_returns.items()
        },
    }


def _run_backtest(market: MarketDataService, body: BacktestRequest) -> dict[str, Any]:
    ticker = body.ticker.upper()
    try:
        df = market.history(ticker)
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    max_splits = max(1, (len(df) - body.train_size) // body.step_size)
    n_splits = min(body.n_splits, max_splits)
    signal = WalkForwardLinearSignal(body.horizon)
    engine = BacktestEngine({ticker: df}, signal, initial_equity=body.initial_equity)
    results = engine.run(
        train_size=body.train_size,
        step_size=body.step_size,
        n_splits=n_splits,
        hold_days=HORIZON_DAYS[body.horizon],
    )
    np.random.seed(settings.model.seed)
    mc = engine.monte_carlo(n_simulations=500, n_years=1, results=results) if len(results.equity_curve) >= 20 else {}
    curve = results.equity_curve
    stride = max(1, len(curve) // 120)
    return {
        "ticker": ticker,
        "horizon": body.horizon,
        "data_mode": "bundled_synthetic_fixture" if market.provider == "bundled" else "external_market_data",
        "method": "walk_forward_ridge_linear",
        "stats": results.stats,
        "monte_carlo": mc,
        "equity_curve": [
            {"date": dt.isoformat(), "equity": round(float(value), 2)}
            for dt, value in curve.iloc[::stride].items()
        ],
        "warnings": [
            "Bundled fixture results validate software behaviour only; they are not empirical trading evidence."
        ] if market.provider == "bundled" else [],
    }


def _run_experiment(market: MarketDataService, body: ExperimentRequest) -> dict[str, Any]:
    ticker = body.ticker.upper()
    try:
        df = market.history(ticker)
    except MarketDataError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    allowed = {item["id"] for item in MODEL_CATALOG}
    selected = list(dict.fromkeys(m.lower() for m in body.models))
    invalid = [m for m in selected if m not in allowed]
    if invalid:
        raise HTTPException(status_code=422, detail=f"unsupported models: {', '.join(invalid)}")
    max_splits = max(1, (len(df) - body.train_size) // body.step_size)
    n_splits = min(body.n_splits, max_splits)
    comparisons = []
    for model in selected:
        engine = BacktestEngine(
            {ticker: df},
            build_signal(model, body.horizon),
            initial_equity=body.initial_equity,
            commission_pct=body.commission_bps / 10_000.0,
            slippage_pct=body.slippage_bps / 10_000.0,
        )
        results = engine.run(
            train_size=body.train_size,
            step_size=body.step_size,
            n_splits=n_splits,
            hold_days=HORIZON_DAYS[body.horizon],
        )
        curve = results.equity_curve
        stride = max(1, len(curve) // 80)
        comparisons.append({
            "model": model,
            "kind": "machine_learning" if model == "ridge" else "baseline",
            "stats": results.stats,
            "trades": len(results.trades),
            "equity_curve": [
                {"date": dt.isoformat(), "equity": round(float(value), 2)}
                for dt, value in curve.iloc[::stride].items()
            ],
        })
    return {
        "ticker": ticker,
        "horizon": body.horizon,
        "data_mode": "bundled_synthetic_fixture" if market.provider == "bundled" else "external_market_data",
        "costs": {"commission_bps": body.commission_bps, "slippage_bps": body.slippage_bps},
        "methodology": "identical walk-forward folds and execution assumptions across models",
        "comparisons": comparisons,
        "warning": "Fixture comparisons demonstrate evaluation plumbing, not real-market alpha." if market.provider == "bundled" else None,
    }


@app.get("/api/v1/intelligence/search", tags=["live-intelligence"])
async def intelligence_search(
    request: Request,
    q: str,
    limit: int = 8,
    user: dict = Depends(current_user),
):
    service: LiveIntelligenceService = request.app.state.intelligence
    if not service.configured:
        raise HTTPException(status_code=503, detail="Live intelligence requires ALPHA_VANTAGE_KEY")
    try:
        matches = await service.search(q, limit=max(1, min(limit, 20)))
    except ProviderLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LiveIntelligenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"query": q, "matches": matches, "count": len(matches)}


@app.post("/api/v1/intelligence/analyze", tags=["live-intelligence"])
async def intelligence_analyze(
    body: LiveIntelligenceRequest,
    request: Request,
    user: dict = Depends(current_user),
):
    service: LiveIntelligenceService = request.app.state.intelligence
    if not service.configured:
        raise HTTPException(status_code=503, detail="Live intelligence requires ALPHA_VANTAGE_KEY")
    cache_key = f"intel:v3:{body.query.strip().lower()}:{body.horizon}:{int(body.include_reddit_comments)}"
    cached = await get_json(request.app.state.cache, cache_key)
    if cached:
        cached["cache_hit"] = True
        return cached
    try:
        payload = await service.analyze(body.query, body.horizon, body.include_reddit_comments)
    except ProviderLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LiveIntelligenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    payload["cache_hit"] = False
    await set_json(request.app.state.cache, cache_key, payload, settings.infra.redis_ttl_intelligence)
    return payload


@app.get("/api/v1/experiments/models", tags=["experiments"])
async def experiment_models(user: dict = Depends(current_user)):
    return {"models": MODEL_CATALOG}


@app.post("/api/v1/experiments/compare", tags=["experiments"])
async def experiment_compare(body: ExperimentRequest, request: Request, user: dict = Depends(current_user)):
    market: MarketDataService = request.app.state.market
    return await asyncio.to_thread(_run_experiment, market, body)


@app.get("/api/v1/system", tags=["system"])
async def system_overview(request: Request, user: dict = Depends(current_user)):
    runtime = request.app.state.runtime
    predictor: PredictionService = request.app.state.predictor
    uptime = max(0.0, time.time() - runtime["started_at"])
    requests_total = int(runtime["requests_total"])
    prediction_requests = int(runtime["prediction_requests"])
    cache_hits = int(runtime["prediction_cache_hits"])
    return {
        "service": "quant-ai-api",
        "version": "3.0.1",
        "uptime_seconds": round(uptime, 2),
        "requests_total": requests_total,
        "average_http_latency_ms": round((runtime["http_seconds_sum"] / requests_total * 1000.0), 3) if requests_total else 0.0,
        "prediction_requests": prediction_requests,
        "prediction_cache_hit_rate": round(cache_hits / prediction_requests, 4) if prediction_requests else 0.0,
        "cache_backend": request.app.state.cache_backend,
        "models_loaded": len(predictor.registry.available()),
        "data_provider": predictor.market.provider,
        "live_intelligence_configured": bool(request.app.state.intelligence.configured),
        "reddit_configured": bool(request.app.state.intelligence.reddit.configured),
        "alpha_vantage_outputsize": settings.data.alpha_vantage_outputsize,
        "provider_cache_ttls_seconds": {
            "symbol_resolution": settings.infra.redis_ttl_symbol_resolution,
            "market_history": settings.infra.redis_ttl_market_history,
            "news": settings.infra.redis_ttl_news,
            "reddit": settings.infra.redis_ttl_reddit,
            "final_intelligence": settings.infra.redis_ttl_intelligence,
        },
        "architecture": ["TypeScript UI", "FastAPI", "Live provider adapters", request.app.state.cache_backend.title() + " cache", "On-demand fusion model"],
        "capabilities": [
            "JWT authentication", "typed REST contracts", "WebSocket streaming", "persisted inference",
            "walk-forward evaluation", "model-vs-baseline experiments", "live company/ticker resolution",
            "price + news model with bounded Reddit inference overlay", "on-demand model fitting", "source-attributed evidence",
            "Prometheus metrics", "container deployment",
        ],
    }


@app.post("/api/v1/backtest", tags=["backtesting"])
async def backtest(body: BacktestRequest, request: Request, user: dict = Depends(current_user)):
    market: MarketDataService = request.app.state.market
    return await asyncio.to_thread(_run_backtest, market, body)


@app.get("/api/v1/backtest/{ticker}", tags=["backtesting"])
async def backtest_get(ticker: str, request: Request, horizon: str = "5d", user: dict = Depends(current_user)):
    if horizon not in HORIZON_DAYS:
        raise HTTPException(status_code=422, detail="horizon must be one of 1d, 5d, 20d")
    market: MarketDataService = request.app.state.market
    return await asyncio.to_thread(_run_backtest, market, BacktestRequest(ticker=ticker, horizon=horizon))


@app.get("/api/v1/backtest/{ticker}/equity.csv", tags=["backtesting"])
async def backtest_csv(ticker: str, request: Request, horizon: str = "5d", user: dict = Depends(current_user)):
    if horizon not in HORIZON_DAYS:
        raise HTTPException(status_code=422, detail="horizon must be one of 1d, 5d, 20d")
    market: MarketDataService = request.app.state.market
    payload = await asyncio.to_thread(_run_backtest, market, BacktestRequest(ticker=ticker, horizon=horizon))
    rows = ["date,equity"] + [f"{row['date']},{row['equity']:.2f}" for row in payload["equity_curve"]]
    content = "\n".join(rows) + "\n"
    headers = {"Content-Disposition": f'attachment; filename="{ticker.upper()}_{horizon}_equity.csv"'}
    return Response(content, media_type="text/csv; charset=utf-8", headers=headers)


async def _ws_auth(ws: WebSocket) -> bool:
    if not api_cfg.auth_required:
        return True
    token = ws.query_params.get("token", "")
    try:
        decode_access_token(token, api_cfg.jwt_secret, api_cfg.jwt_algorithm)
        return True
    except jwt.PyJWTError:
        await ws.close(code=1008, reason="invalid token")
        return False


@app.websocket("/ws/predictions/{ticker}")
async def ws_predictions(ws: WebSocket, ticker: str):
    if not await _ws_auth(ws):
        return
    await ws.accept()
    predictor = ws.app.state.predictor
    horizon = ws.query_params.get("horizon", settings.model.default_horizon)
    if horizon not in HORIZON_DAYS:
        horizon = settings.model.default_horizon
    try:
        while True:
            try:
                payload = predictor.predict(ticker, horizon)
                payload.update({"type": "prediction_update"})
                await ws.send_json(payload)
            except Exception as exc:
                await ws.send_json({"type": "error", "detail": str(exc)})
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=settings.trading.price_poll_interval)
                if text == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        return


@app.websocket("/ws/market/feed")
async def ws_market_feed(ws: WebSocket):
    if not await _ws_auth(ws):
        return
    await ws.accept()
    predictor: PredictionService = ws.app.state.predictor
    tickers = predictor.market.available_tickers()[:5]
    try:
        while True:
            payloads = []
            for ticker in tickers:
                try:
                    payloads.append(predictor.predict(ticker, settings.model.default_horizon))
                except Exception:
                    continue
            await ws.send_json({
                "type": "market_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_mode": "bundled_synthetic_fixture" if predictor.market.provider == "bundled" else "external_market_data",
                "predictions": payloads,
            })
            await asyncio.sleep(settings.trading.price_poll_interval)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=api_cfg.host, port=api_cfg.port, workers=api_cfg.workers)
