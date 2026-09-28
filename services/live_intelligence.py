"""Live market-intelligence pipeline.

The module deliberately uses documented provider APIs instead of web scraping:
- Alpha Vantage: symbol resolution, daily OHLCV, market news + provider sentiment.
- Reddit Data API: OAuth search over public posts and optional top-thread comments.

Reddit text is processed ephemerally for inference/aggregation. The service does
not persist Reddit content or train model weights on Reddit content.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import httpx
import numpy as np
import pandas as pd

from services.cache import get_json, set_json
from services.prediction_service import FEATURES as TECHNICAL_FEATURES
from services.prediction_service import HORIZON_DAYS, feature_frame


class LiveIntelligenceError(RuntimeError):
    """Raised when a live provider cannot satisfy an intelligence request."""


class ProviderLimitError(LiveIntelligenceError):
    """Provider quota/rate-limit/premium boundary with a machine-readable kind."""

    def __init__(self, message: str, kind: str = "quota"):
        super().__init__(message)
        self.kind = kind


@dataclass(slots=True)
class StockCandidate:
    symbol: str
    name: str
    asset_type: str = "Equity"
    region: str = ""
    market_open: str = ""
    market_close: str = ""
    timezone: str = ""
    currency: str = ""
    match_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_type": self.asset_type,
            "region": self.region,
            "currency": self.currency,
            "match_score": round(self.match_score, 4),
        }


@dataclass(slots=True)
class IntelligenceItem:
    source: str
    kind: str
    title: str
    text: str
    published_at: datetime
    url: str
    sentiment: float
    relevance: float = 1.0
    engagement: int = 0
    subreddit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        # Do not proxy full third-party bodies through the portfolio app.
        snippet = re.sub(r"\s+", " ", self.text or "").strip()[:280]
        return {
            "source": self.source,
            "kind": self.kind,
            "title": self.title[:220],
            "snippet": snippet,
            "published_at": self.published_at.astimezone(timezone.utc).isoformat(),
            "url": self.url,
            "sentiment": round(float(self.sentiment), 4),
            "relevance": round(float(self.relevance), 4),
            "engagement": int(self.engagement),
            "subreddit": self.subreddit,
        }


class FinancialLexiconSentiment:
    """Small deterministic fallback for social text.

    Alpha Vantage's ticker-level provider sentiment is preferred for news.
    This scorer exists so Reddit inference remains runnable without downloading
    a large transformer model. It is intentionally transparent and replaceable.
    """

    POSITIVE = {
        "beat": 1.2, "beats": 1.2, "upgrade": 1.1, "upgraded": 1.1,
        "growth": 0.8, "profit": 0.9, "profitable": 1.0, "bullish": 1.2,
        "outperform": 1.1, "strong": 0.7, "surge": 1.0, "surges": 1.0,
        "rally": 0.9, "gain": 0.7, "gains": 0.7, "record": 0.6,
        "buyback": 0.8, "approval": 0.8, "approved": 0.8, "contract": 0.5,
        "partnership": 0.5, "expansion": 0.5, "demand": 0.4, "momentum": 0.5,
        "rebound": 0.7, "recovery": 0.6, "raises": 0.7, "raised": 0.7,
        "optimistic": 0.8, "innovation": 0.4, "breakout": 0.8, "buy": 0.5,
        "long": 0.35, "moon": 0.6, "mooning": 0.7,
    }
    NEGATIVE = {
        "miss": -1.1, "misses": -1.1, "downgrade": -1.1, "downgraded": -1.1,
        "loss": -0.9, "losses": -0.9, "bearish": -1.2, "underperform": -1.1,
        "weak": -0.7, "plunge": -1.0, "plunges": -1.0, "fall": -0.6,
        "falls": -0.6, "drop": -0.6, "drops": -0.6, "lawsuit": -0.8,
        "probe": -0.7, "investigation": -0.8, "recall": -0.7,
        "bankruptcy": -1.4, "bankrupt": -1.4, "dilution": -0.9,
        "fraud": -1.3, "layoff": -0.6, "layoffs": -0.6, "warning": -0.7,
        "cut": -0.6, "cuts": -0.6, "slump": -0.8, "crash": -1.2,
        "selloff": -1.0, "risk": -0.35, "uncertainty": -0.45,
        "disappointing": -0.8, "missed": -1.0, "sell": -0.5, "short": -0.35,
    }
    NEGATIONS = {"not", "no", "never", "without", "isn't", "wasn't", "won't", "dont", "don't"}
    TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'_-]*")

    def score(self, text: str) -> float:
        tokens = [t.lower() for t in self.TOKEN_RE.findall(text or "")]
        if not tokens:
            return 0.0
        total = 0.0
        hits = 0
        for idx, token in enumerate(tokens):
            value = self.POSITIVE.get(token, self.NEGATIVE.get(token, 0.0))
            if not value:
                continue
            if any(t in self.NEGATIONS for t in tokens[max(0, idx - 3):idx]):
                value *= -0.75
            total += value
            hits += 1
        if not hits:
            return 0.0
        # Smooth saturation makes long posts comparable with short headlines.
        return float(np.tanh(total / max(1.0, math.sqrt(hits))))


class AlphaVantageClient:
    BASE = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        outputsize: str = "compact",
        min_interval_seconds: float = 1.10,
    ):
        self.api_key = api_key.strip()
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0))
        self.sentiment = FinancialLexiconSentiment()
        self.outputsize = "full" if str(outputsize).lower() == "full" else "compact"
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _get(self, **params: Any) -> dict[str, Any]:
        if not self.configured:
            raise LiveIntelligenceError("ALPHA_VANTAGE_KEY is required for live stock intelligence")
        params["apikey"] = self.api_key

        # Alpha Vantage's free tier is intentionally paced. Serialising calls
        # here prevents one analysis from violating the per-second burst limit.
        async with self._request_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait_for = self.min_interval_seconds - elapsed
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            try:
                response = await self.client.get(self.BASE, params=params)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise LiveIntelligenceError(f"Alpha Vantage request failed: {exc}") from exc
            finally:
                self._last_request_at = time.monotonic()

        if payload.get("Error Message"):
            raise LiveIntelligenceError(str(payload["Error Message"]))
        provider_message = str(payload.get("Note") or payload.get("Information") or "").strip()
        if provider_message:
            lower = provider_message.lower()
            if "25 requests per day" in lower or "daily" in lower and "limit" in lower:
                raise ProviderLimitError(
                    "Alpha Vantage daily quota reached. Cached results and the Offline Lab still work; try live analysis again after the provider quota resets.",
                    kind="daily_quota",
                )
            if "request per second" in lower or "rate limit" in lower:
                raise ProviderLimitError(
                    "Alpha Vantage temporary rate limit reached. Wait a few seconds and retry; the app now spaces provider calls automatically.",
                    kind="burst_rate",
                )
            if "premium" in lower:
                raise ProviderLimitError(
                    "Alpha Vantage reports that this request needs a premium capability. The free-tier configuration should use ALPHA_VANTAGE_OUTPUTSIZE=compact.",
                    kind="premium",
                )
            raise LiveIntelligenceError(provider_message)
        return payload

    async def search(self, query: str, limit: int = 8) -> list[StockCandidate]:
        payload = await self._get(function="SYMBOL_SEARCH", keywords=query)
        matches: list[StockCandidate] = []
        for row in payload.get("bestMatches", []):
            try:
                score = float(row.get("9. matchScore", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            candidate = StockCandidate(
                symbol=str(row.get("1. symbol", "")).upper(),
                name=str(row.get("2. name", "")),
                asset_type=str(row.get("3. type", "")),
                region=str(row.get("4. region", "")),
                market_open=str(row.get("5. marketOpen", "")),
                market_close=str(row.get("6. marketClose", "")),
                timezone=str(row.get("7. timezone", "")),
                currency=str(row.get("8. currency", "")),
                match_score=score,
            )
            if candidate.symbol:
                matches.append(candidate)
        # Equity/ETF-like instruments first, then provider relevance.
        matches.sort(
            key=lambda item: (
                1 if any(k in item.asset_type.lower() for k in ("equity", "etf", "stock")) else 0,
                item.match_score,
            ),
            reverse=True,
        )
        return matches[: max(1, min(limit, 20))]

    async def resolve(self, query: str) -> StockCandidate:
        matches = await self.search(query, limit=10)
        if not matches:
            raise LiveIntelligenceError(f"No listed instrument matched '{query}'")
        cleaned = query.strip().upper()
        for match in matches:
            if match.symbol == cleaned:
                return match
        return matches[0]

    async def daily_history(self, symbol: str, max_rows: int = 1500) -> pd.DataFrame:
        payload = await self._get(function="TIME_SERIES_DAILY", symbol=symbol.upper(), outputsize=self.outputsize)
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            raise LiveIntelligenceError(f"No daily history returned for {symbol.upper()}")
        records: list[dict[str, Any]] = []
        for date, row in series.items():
            try:
                records.append({
                    "date": pd.Timestamp(date),
                    "open": float(row["1. open"]),
                    "high": float(row["2. high"]),
                    "low": float(row["3. low"]),
                    "close": float(row["4. close"]),
                    "volume": float(row["5. volume"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        if not records:
            raise LiveIntelligenceError(f"Daily history for {symbol.upper()} could not be parsed")
        df = pd.DataFrame(records).set_index("date").sort_index().tail(max_rows)
        return df

    async def news(self, symbol: str, lookback_days: int = 365, limit: int = 500) -> list[IntelligenceItem]:
        time_from = (datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))).strftime("%Y%m%dT%H%M")
        payload = await self._get(
            function="NEWS_SENTIMENT",
            tickers=symbol.upper(),
            time_from=time_from,
            sort="LATEST",
            limit=max(1, min(limit, 1000)),
        )
        items: list[IntelligenceItem] = []
        for article in payload.get("feed", []) or []:
            title = str(article.get("title", ""))
            summary = str(article.get("summary", ""))
            raw_time = str(article.get("time_published", ""))
            try:
                published = datetime.strptime(raw_time[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                published = datetime.now(timezone.utc)

            ticker_score: float | None = None
            relevance = 0.0
            for ticker_row in article.get("ticker_sentiment", []) or []:
                if str(ticker_row.get("ticker", "")).upper() == symbol.upper():
                    try:
                        ticker_score = float(ticker_row.get("ticker_sentiment_score", 0.0))
                    except (TypeError, ValueError):
                        ticker_score = None
                    try:
                        relevance = float(ticker_row.get("relevance_score", 0.0))
                    except (TypeError, ValueError):
                        relevance = 0.0
                    break
            sentiment = ticker_score if ticker_score is not None else self.sentiment.score(f"{title}. {summary}")
            items.append(IntelligenceItem(
                source=str(article.get("source", "News")) or "News",
                kind="news",
                title=title,
                text=summary,
                published_at=published,
                url=str(article.get("url", "")),
                sentiment=float(np.clip(sentiment, -1.0, 1.0)),
                relevance=float(np.clip(relevance or 0.5, 0.0, 1.0)),
                metadata={"provider": "alpha_vantage"},
            ))
        return items


class RedditClient:
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    OAUTH_BASE = "https://oauth.reddit.com"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
    ):
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.user_agent = user_agent.strip() or "QuantAIResearch/3.0"
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0))
        self._access_token = ""
        self._token_expires_at = 0.0
        self.sentiment = FinancialLexiconSentiment()

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _token(self) -> str:
        if not self.configured:
            raise LiveIntelligenceError("Reddit credentials are not configured")
        if self._access_token and time.time() < self._token_expires_at - 30:
            return self._access_token
        try:
            response = await self.client.post(
                self.TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                headers={"User-Agent": self.user_agent},
                data={"grant_type": "client_credentials"},
            )
            response.raise_for_status()
            payload = response.json()
            token = str(payload.get("access_token", ""))
            if not token:
                raise LiveIntelligenceError("Reddit OAuth response did not include an access token")
            self._access_token = token
            self._token_expires_at = time.time() + float(payload.get("expires_in", 3600))
            return token
        except (httpx.HTTPError, ValueError) as exc:
            raise LiveIntelligenceError(f"Reddit OAuth failed: {exc}") from exc

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        token = await self._token()
        try:
            response = await self.client.get(
                f"{self.OAUTH_BASE}{path}",
                params=params,
                headers={"Authorization": f"bearer {token}", "User-Agent": self.user_agent},
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LiveIntelligenceError(f"Reddit API request failed: {exc}") from exc

    async def search_posts(self, ticker: str, company_name: str, limit: int = 100) -> list[IntelligenceItem]:
        if not self.configured:
            return []
        query = f'"{company_name}" OR {ticker.upper()} OR ${ticker.upper()}' if company_name else ticker.upper()
        target = max(1, min(limit, 200))
        items: list[IntelligenceItem] = []
        seen: set[str] = set()
        after: str | None = None
        while len(items) < target:
            batch = min(100, target - len(items))
            params: dict[str, Any] = {
                "q": query,
                "sort": "new",
                "t": "year",
                "limit": batch,
                "type": "link",
                "raw_json": 1,
            }
            if after:
                params["after"] = after
            payload = await self._get("/search", params)
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            children = data.get("children", []) or []
            for child in children:
                row = child.get("data", {}) if isinstance(child, dict) else {}
                post_id = str(row.get("id", ""))
                if not post_id or post_id in seen:
                    continue
                seen.add(post_id)
                title = str(row.get("title", ""))
                selftext = str(row.get("selftext", ""))
                created = datetime.fromtimestamp(float(row.get("created_utc", time.time())), tz=timezone.utc)
                score = int(row.get("score", 0) or 0)
                comments = int(row.get("num_comments", 0) or 0)
                permalink = str(row.get("permalink", ""))
                items.append(IntelligenceItem(
                    source="Reddit",
                    kind="reddit_post",
                    title=title,
                    text=selftext,
                    published_at=created,
                    url=f"https://www.reddit.com{permalink}" if permalink else "",
                    sentiment=self.sentiment.score(f"{title}. {selftext}"),
                    relevance=1.0,
                    engagement=max(0, score) + max(0, comments),
                    subreddit=str(row.get("subreddit", "")),
                    metadata={"id": post_id, "score": score, "num_comments": comments},
                ))
            after = data.get("after")
            if not after or not children:
                break
        return items[:target]

    async def top_comments(self, posts: list[IntelligenceItem], max_posts: int = 3, comments_per_post: int = 15) -> list[IntelligenceItem]:
        if not self.configured:
            return []
        selected = sorted(posts, key=lambda item: item.engagement, reverse=True)[: max(0, max_posts)]
        comments: list[IntelligenceItem] = []
        for post in selected:
            post_id = str(post.metadata.get("id", ""))
            if not post_id:
                continue
            try:
                payload = await self._get(
                    f"/comments/{post_id}",
                    {"sort": "top", "limit": max(1, min(comments_per_post, 50)), "depth": 1, "raw_json": 1},
                )
            except LiveIntelligenceError:
                continue
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            listing = payload[1].get("data", {}) if isinstance(payload[1], dict) else {}
            for child in listing.get("children", []) or []:
                if child.get("kind") != "t1":
                    continue
                row = child.get("data", {})
                body = str(row.get("body", ""))
                if not body or body in {"[deleted]", "[removed]"}:
                    continue
                created = datetime.fromtimestamp(float(row.get("created_utc", time.time())), tz=timezone.utc)
                permalink = str(row.get("permalink", ""))
                comments.append(IntelligenceItem(
                    source="Reddit",
                    kind="reddit_comment",
                    title=f"Comment in: {post.title}"[:220],
                    text=body,
                    published_at=created,
                    url=f"https://www.reddit.com{permalink}" if permalink else post.url,
                    sentiment=self.sentiment.score(body),
                    relevance=1.0,
                    engagement=max(0, int(row.get("score", 0) or 0)),
                    subreddit=post.subreddit,
                    metadata={"parent_post": post_id},
                ))
        return comments


ALT_FEATURES = [
    "news_sentiment",
    "news_volume",
    "reddit_sentiment",
    "reddit_volume",
    "reddit_engagement",
]


class LiveIntelligenceService:
    """Resolve a company, collect live evidence and fit an on-demand fusion model."""

    def __init__(
        self,
        alpha: AlphaVantageClient,
        reddit: RedditClient,
        lookback_days: int = 365,
        news_fetch_limit: int = 500,
        reddit_fetch_limit: int = 100,
        reddit_comment_posts: int = 5,
        reddit_comments_per_post: int = 20,
        cache: Any | None = None,
        symbol_cache_ttl: int = 604800,
        history_cache_ttl: int = 21600,
        news_cache_ttl: int = 1800,
        reddit_cache_ttl: int = 600,
    ):
        self.alpha = alpha
        self.reddit = reddit
        self.lookback_days = max(30, min(lookback_days, 3650))
        self.news_fetch_limit = max(20, min(news_fetch_limit, 1000))
        self.reddit_fetch_limit = max(10, min(reddit_fetch_limit, 200))
        self.reddit_comment_posts = max(0, min(reddit_comment_posts, 10))
        self.reddit_comments_per_post = max(1, min(reddit_comments_per_post, 50))
        self.cache = cache
        self.symbol_cache_ttl = max(60, int(symbol_cache_ttl))
        self.history_cache_ttl = max(60, int(history_cache_ttl))
        self.news_cache_ttl = max(60, int(news_cache_ttl))
        self.reddit_cache_ttl = max(60, int(reddit_cache_ttl))

    @property
    def configured(self) -> bool:
        return self.alpha.configured

    async def close(self) -> None:
        await asyncio.gather(self.alpha.close(), self.reddit.close())

    @staticmethod
    def _normalise_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip()).lower()

    @staticmethod
    def _looks_like_explicit_ticker(query: str) -> bool:
        raw = query.strip()
        # Optimisation is intentionally conservative: only clearly ticker-like
        # uppercase inputs skip SYMBOL_SEARCH. "Apple" still resolves normally.
        return bool(raw and raw == raw.upper() and re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", raw))

    async def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.cache is None:
            return None
        try:
            return await get_json(self.cache, key)
        except Exception:
            return None

    async def _cache_set(self, key: str, value: dict[str, Any], ttl: int) -> None:
        if self.cache is None:
            return
        try:
            await set_json(self.cache, key, value, ttl)
        except Exception:
            return

    @staticmethod
    def _candidate_from_cache(data: dict[str, Any]) -> StockCandidate:
        return StockCandidate(
            symbol=str(data.get("symbol", "")),
            name=str(data.get("name", "")),
            asset_type=str(data.get("asset_type", "Equity")),
            region=str(data.get("region", "")),
            market_open=str(data.get("market_open", "")),
            market_close=str(data.get("market_close", "")),
            timezone=str(data.get("timezone", "")),
            currency=str(data.get("currency", "")),
            match_score=float(data.get("match_score", 0.0) or 0.0),
        )

    @staticmethod
    def _candidate_cache_dict(item: StockCandidate) -> dict[str, Any]:
        return {
            "symbol": item.symbol, "name": item.name, "asset_type": item.asset_type,
            "region": item.region, "market_open": item.market_open, "market_close": item.market_close,
            "timezone": item.timezone, "currency": item.currency, "match_score": item.match_score,
        }

    @staticmethod
    def _item_cache_dict(item: IntelligenceItem) -> dict[str, Any]:
        return {
            "source": item.source, "kind": item.kind, "title": item.title, "text": item.text,
            "published_at": item.published_at.astimezone(timezone.utc).isoformat(), "url": item.url,
            "sentiment": item.sentiment, "relevance": item.relevance, "engagement": item.engagement,
            "subreddit": item.subreddit, "metadata": item.metadata,
        }

    @staticmethod
    def _item_from_cache(data: dict[str, Any]) -> IntelligenceItem:
        published_raw = str(data.get("published_at", ""))
        try:
            published = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
        except ValueError:
            published = datetime.now(timezone.utc)
        return IntelligenceItem(
            source=str(data.get("source", "")), kind=str(data.get("kind", "")),
            title=str(data.get("title", "")), text=str(data.get("text", "")), published_at=published,
            url=str(data.get("url", "")), sentiment=float(data.get("sentiment", 0.0) or 0.0),
            relevance=float(data.get("relevance", 1.0) or 0.0), engagement=int(data.get("engagement", 0) or 0),
            subreddit=str(data.get("subreddit", "")), metadata=dict(data.get("metadata", {}) or {}),
        )

    async def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        key = f"provider:v3:symbol-search:{self._normalise_query(query)}"
        cached = await self._cache_get(key)
        if cached and isinstance(cached.get("matches"), list):
            return list(cached["matches"])[: max(1, min(limit, 20))]
        matches = await self.alpha.search(query, limit=max(limit, 10))
        payload = [self._candidate_cache_dict(item) for item in matches]
        await self._cache_set(key, {"matches": payload}, self.symbol_cache_ttl)
        return [item.as_dict() for item in matches[: max(1, min(limit, 20))]]

    async def _resolve(self, query: str) -> StockCandidate:
        raw = query.strip()
        if self._looks_like_explicit_ticker(raw):
            return StockCandidate(symbol=raw.upper(), name=raw.upper(), match_score=1.0)
        key = f"provider:v3:resolved:{self._normalise_query(raw)}"
        cached = await self._cache_get(key)
        if cached and cached.get("candidate"):
            return self._candidate_from_cache(dict(cached["candidate"]))
        matches = await self.alpha.search(raw, limit=10)
        if not matches:
            raise LiveIntelligenceError(f"No listed instrument matched '{raw}'")
        cleaned = raw.upper()
        candidate = next((item for item in matches if item.symbol == cleaned), matches[0])
        await self._cache_set(key, {"candidate": self._candidate_cache_dict(candidate)}, self.symbol_cache_ttl)
        return candidate

    async def _history(self, symbol: str) -> pd.DataFrame:
        key = f"provider:v3:history:{symbol.upper()}:{getattr(self.alpha, 'outputsize', 'compact')}"
        cached = await self._cache_get(key)
        if cached and isinstance(cached.get("rows"), list) and cached["rows"]:
            df = pd.DataFrame(cached["rows"])
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").sort_index()
        df = await self.alpha.daily_history(symbol)
        rows = [
            {"date": idx.date().isoformat(), "open": float(row["open"]), "high": float(row["high"]),
             "low": float(row["low"]), "close": float(row["close"]), "volume": float(row["volume"])}
            for idx, row in df.iterrows()
        ]
        await self._cache_set(key, {"rows": rows}, self.history_cache_ttl)
        return df

    async def _news(self, symbol: str, lookback_days: int) -> list[IntelligenceItem]:
        key = f"provider:v3:news:{symbol.upper()}:{lookback_days}:{self.news_fetch_limit}"
        cached = await self._cache_get(key)
        if cached and isinstance(cached.get("items"), list):
            return [self._item_from_cache(dict(item)) for item in cached["items"]]
        items = await self.alpha.news(symbol, lookback_days, self.news_fetch_limit)
        await self._cache_set(key, {"items": [self._item_cache_dict(item) for item in items]}, self.news_cache_ttl)
        return items

    async def _reddit_posts(self, symbol: str, company_name: str) -> list[IntelligenceItem]:
        if not self.reddit.configured:
            return []
        key = f"provider:v3:reddit-posts:{symbol.upper()}:{self.reddit_fetch_limit}"
        cached = await self._cache_get(key)
        if cached and isinstance(cached.get("items"), list):
            return [self._item_from_cache(dict(item)) for item in cached["items"]]
        items = await self.reddit.search_posts(symbol, company_name, self.reddit_fetch_limit)
        await self._cache_set(key, {"items": [self._item_cache_dict(item) for item in items]}, self.reddit_cache_ttl)
        return items

    async def _reddit_comments(self, posts: list[IntelligenceItem]) -> list[IntelligenceItem]:
        if not self.reddit.configured or not posts or self.reddit_comment_posts <= 0:
            return []
        ids = ",".join(str(item.metadata.get("id", "")) for item in sorted(posts, key=lambda x: x.engagement, reverse=True)[:self.reddit_comment_posts])
        key = f"provider:v3:reddit-comments:{ids}:{self.reddit_comments_per_post}"
        cached = await self._cache_get(key)
        if cached and isinstance(cached.get("items"), list):
            return [self._item_from_cache(dict(item)) for item in cached["items"]]
        items = await self.reddit.top_comments(posts, max_posts=self.reddit_comment_posts, comments_per_post=self.reddit_comments_per_post)
        await self._cache_set(key, {"items": [self._item_cache_dict(item) for item in items]}, self.reddit_cache_ttl)
        return items

    @staticmethod
    def _weighted_source_summary(items: list[IntelligenceItem]) -> dict[str, Any]:
        if not items:
            return {"score": 0.0, "volume": 0, "engagement": 0, "positive": 0, "negative": 0, "neutral": 0}
        scores = []
        weights = []
        for item in items:
            weight = max(0.05, item.relevance) * (1.0 + math.log1p(max(0, item.engagement)) / 5.0)
            scores.append(float(item.sentiment))
            weights.append(weight)
        score = float(np.average(scores, weights=weights)) if sum(weights) else 0.0
        return {
            "score": round(float(np.clip(score, -1.0, 1.0)), 4),
            "volume": len(items),
            "engagement": int(sum(max(0, item.engagement) for item in items)),
            "positive": sum(1 for s in scores if s > 0.15),
            "negative": sum(1 for s in scores if s < -0.15),
            "neutral": sum(1 for s in scores if -0.15 <= s <= 0.15),
        }

    @staticmethod
    def _map_items_to_market_dates(index: pd.DatetimeIndex, items: Iterable[IntelligenceItem]) -> pd.DataFrame:
        dates = pd.DatetimeIndex(index).tz_localize(None).normalize()
        frame = pd.DataFrame(index=dates, data={
            "news_sentiment": 0.0,
            "news_volume": 0.0,
            "reddit_sentiment": 0.0,
            "reddit_volume": 0.0,
            "reddit_engagement": 0.0,
        })
        if len(frame) == 0:
            return frame

        buckets: dict[tuple[pd.Timestamp, str], list[tuple[float, float, int]]] = {}
        for item in items:
            day = pd.Timestamp(item.published_at.astimezone(timezone.utc).date())
            # Content outside the available market-history window is not
            # collapsed onto the first/last training row. This prevents old
            # news from contaminating the compact free-tier history window.
            if day < dates[0] or day > dates[-1]:
                continue
            pos = dates.searchsorted(day, side="left")
            if pos >= len(dates):
                continue
            market_day = dates[pos]
            source = "news" if item.kind == "news" else "reddit"
            weight = max(0.05, item.relevance) * (1.0 + math.log1p(max(0, item.engagement)) / 5.0)
            buckets.setdefault((market_day, source), []).append((item.sentiment, weight, item.engagement))

        for (day, source), values in buckets.items():
            scores = np.asarray([v[0] for v in values], dtype=float)
            weights = np.asarray([v[1] for v in values], dtype=float)
            score = float(np.average(scores, weights=weights)) if weights.sum() else 0.0
            if source == "news":
                frame.loc[day, "news_sentiment"] = score
                frame.loc[day, "news_volume"] = math.log1p(len(values))
            else:
                frame.loc[day, "reddit_sentiment"] = score
                frame.loc[day, "reddit_volume"] = math.log1p(len(values))
                frame.loc[day, "reddit_engagement"] = math.log1p(sum(max(0, v[2]) for v in values))

        # Preserve short-lived information while letting stale sentiment decay.
        for col in ALT_FEATURES:
            frame[col] = frame[col].ewm(span=5, adjust=False).mean()
        return frame

    @staticmethod
    def _recent_alt_vector(items: list[IntelligenceItem], now: datetime) -> dict[str, float]:
        cutoff = now - timedelta(days=7)
        recent = [item for item in items if item.published_at >= cutoff]
        news = [item for item in recent if item.kind == "news"]
        reddit = [item for item in recent if item.kind.startswith("reddit_")]
        ns = LiveIntelligenceService._weighted_source_summary(news)
        rs = LiveIntelligenceService._weighted_source_summary(reddit)
        return {
            "news_sentiment": float(ns["score"]),
            "news_volume": math.log1p(int(ns["volume"])),
            "reddit_sentiment": float(rs["score"]),
            "reddit_volume": math.log1p(int(rs["volume"])),
            "reddit_engagement": math.log1p(int(rs["engagement"])),
        }

    @staticmethod
    def _ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float = 0.35) -> dict[str, Any]:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-10] = 1.0
        Z = (X - mean) / std
        design = np.column_stack([np.ones(len(Z)), Z])
        reg = np.eye(design.shape[1]) * ridge
        reg[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
        pred = design @ beta
        sigma = float(np.std(y - pred, ddof=1)) if len(y) > 2 else float(np.std(y - pred))
        return {
            "mean": mean,
            "std": std,
            "intercept": float(beta[0]),
            "coef": beta[1:],
            "residual_std": max(sigma, 1e-6),
        }

    @classmethod
    def _fit_fusion_model(
        cls,
        history: pd.DataFrame,
        items: list[IntelligenceItem],
        horizon: str,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        days = HORIZON_DAYS[horizon]
        technical = feature_frame(history)
        alt = cls._map_items_to_market_dates(history.index, items)
        features = technical.join(alt, how="left").fillna(0.0)
        # Reddit is intentionally excluded from fitted model weights. Reddit's
        # current sentiment is applied later as a small, fixed inference-only
        # overlay so API user content is not used to train an algorithmic model.
        names = list(TECHNICAL_FEATURES) + ["news_sentiment", "news_volume"]
        target = history["close"].shift(-days) / history["close"] - 1.0
        joined = features[names].assign(target=target).replace([np.inf, -np.inf], np.nan).dropna()
        if len(joined) < 60:
            raise LiveIntelligenceError("Not enough clean historical observations to fit an on-demand model")

        n_val = max(20, int(len(joined) * 0.2)) if len(joined) >= 120 else max(10, int(len(joined) * 0.15))
        train = joined.iloc[:-n_val]
        valid = joined.iloc[-n_val:]
        train_fit = cls._ridge_fit(train[names].to_numpy(float), train["target"].to_numpy(float))
        zv = (valid[names].to_numpy(float) - train_fit["mean"]) / train_fit["std"]
        val_pred = train_fit["intercept"] + zv @ train_fit["coef"]
        val_y = valid["target"].to_numpy(float)
        mae = float(np.mean(np.abs(val_y - val_pred)))
        directional = float(np.mean(np.sign(val_y) == np.sign(val_pred)))
        correlation = float(np.corrcoef(val_y, val_pred)[0, 1]) if len(valid) > 2 and np.std(val_pred) > 0 and np.std(val_y) > 0 else 0.0

        final_fit = cls._ridge_fit(joined[names].to_numpy(float), joined["target"].to_numpy(float))
        return final_fit, {"mae": mae, "directional_accuracy": directional, "correlation": correlation}, {
            "feature_names": names,
            "training_rows": len(joined),
            "train_start": joined.index.min().date().isoformat(),
            "train_end": joined.index.max().date().isoformat(),
        }

    @staticmethod
    def _trend_metrics(history: pd.DataFrame) -> dict[str, Any]:
        close = history["close"].astype(float)
        current = float(close.iloc[-1])
        returns: dict[str, float] = {}
        for name, days in (("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60), ("252d", 252)):
            if len(close) > days:
                returns[name] = float((current / float(close.iloc[-days - 1]) - 1.0) * 100.0)
        rolling_252 = close.tail(min(252, len(close)))
        annual_vol = float(close.pct_change().dropna().tail(60).std() * math.sqrt(252) * 100.0)
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else current
        ma50 = float(close.tail(50).mean()) if len(close) >= 50 else ma20
        if current > ma20 > ma50:
            regime = "uptrend"
        elif current < ma20 < ma50:
            regime = "downtrend"
        else:
            regime = "mixed"
        return {
            "current_price": round(current, 4),
            "returns_pct": {k: round(v, 3) for k, v in returns.items()},
            "annualised_volatility_pct": round(annual_vol, 3),
            "high_52w": round(float(rolling_252.max()), 4),
            "low_52w": round(float(rolling_252.min()), 4),
            "distance_to_20d_ma_pct": round((current / ma20 - 1.0) * 100.0, 3),
            "distance_to_50d_ma_pct": round((current / ma50 - 1.0) * 100.0, 3),
            "trend_regime": regime,
            "history_rows": int(len(history)),
            "history_start": history.index.min().date().isoformat(),
            "history_end": history.index.max().date().isoformat(),
        }

    @staticmethod
    def _probabilities(predicted_return: float, sigma: float) -> tuple[str, float, float, float, float]:
        score = float(np.clip(predicted_return / max(sigma, 1e-6), -8.0, 8.0))
        buy = math.exp(score)
        sell = math.exp(-score)
        hold = math.exp(-abs(score) * 0.35 + 0.2)
        total = buy + sell + hold
        pb, ps, ph = buy / total, sell / total, hold / total
        maximum = max(pb, ps, ph)
        if pb == maximum:
            signal = "STRONG_BUY" if pb >= 0.72 else "BUY"
        elif ps == maximum:
            signal = "STRONG_SELL" if ps >= 0.72 else "SELL"
        else:
            signal = "HOLD"
        return signal, maximum, pb, ph, ps

    async def analyze(
        self,
        query: str,
        horizon: str = "5d",
        include_reddit_comments: bool = True,
    ) -> dict[str, Any]:
        if horizon not in HORIZON_DAYS:
            raise LiveIntelligenceError(f"Unsupported horizon '{horizon}'")
        if not self.configured:
            raise LiveIntelligenceError("Live intelligence is not configured; set ALPHA_VANTAGE_KEY")

        started = time.perf_counter()
        company = await self._resolve(query)
        symbol = company.symbol

        # Provider-level caching is deliberately below the final-response cache:
        # switching horizon or Reddit-comment mode should not spend another
        # Alpha Vantage history/news request.
        history = await self._history(symbol)
        history_calendar_days = max(30, (datetime.now(timezone.utc).date() - history.index.min().date()).days + 7)
        effective_news_lookback = min(self.lookback_days, history_calendar_days)
        news_task = asyncio.create_task(self._news(symbol, effective_news_lookback))
        reddit_task = asyncio.create_task(self._reddit_posts(symbol, company.name))
        news_result, reddit_result = await asyncio.gather(news_task, reddit_task, return_exceptions=True)

        source_errors: dict[str, str] = {}
        if isinstance(news_result, Exception):
            source_errors["news"] = str(news_result)
            news: list[IntelligenceItem] = []
        else:
            news = news_result
        if isinstance(reddit_result, Exception):
            source_errors["reddit"] = str(reddit_result)
            reddit_posts: list[IntelligenceItem] = []
        else:
            reddit_posts = reddit_result

        reddit_comments: list[IntelligenceItem] = []
        if include_reddit_comments and reddit_posts and self.reddit.configured:
            try:
                reddit_comments = await self._reddit_comments(reddit_posts)
            except Exception as exc:  # graceful source degradation is intentional
                source_errors["reddit_comments"] = str(exc)

        all_items = news + reddit_posts + reddit_comments
        fit, validation, meta = self._fit_fusion_model(history, news, horizon)
        names: list[str] = meta["feature_names"]

        technical_latest = feature_frame(history).dropna()
        if technical_latest.empty:
            raise LiveIntelligenceError("Technical feature construction returned no usable row")
        current: dict[str, float] = {
            name: float(technical_latest.iloc[-1][name]) for name in TECHNICAL_FEATURES
        }
        current_alt = self._recent_alt_vector(all_items, datetime.now(timezone.utc))
        current.update(current_alt)
        x = np.asarray([current.get(name, 0.0) for name in names], dtype=float)
        z = (x - fit["mean"]) / fit["std"]
        contributions = z * fit["coef"]
        base_pred = float(fit["intercept"] + z @ fit["coef"])
        sigma = float(fit["residual_std"])

        # Inference-only Reddit overlay. The coefficient is deliberately fixed
        # and bounded rather than learned from Reddit content. Higher discussion
        # volume/engagement increases trust in the current social score, but the
        # maximum effect remains a fraction of one residual standard deviation.
        reddit_volume = float(current_alt.get("reddit_volume", 0.0))
        reddit_engagement = float(current_alt.get("reddit_engagement", 0.0))
        reddit_sentiment = float(current_alt.get("reddit_sentiment", 0.0))
        reddit_evidence_strength = float(np.tanh(reddit_volume / 1.5 + reddit_engagement / 8.0))
        reddit_overlay = (
            sigma * 0.25 * reddit_sentiment * (0.25 + 0.75 * reddit_evidence_strength)
            if reddit_volume > 0 else 0.0
        )
        raw_pred = base_pred + reddit_overlay
        caps = {"1d": 0.10, "5d": 0.25, "20d": 0.45}
        predicted_return = float(np.clip(raw_pred, -caps[horizon], caps[horizon]))
        signal, base_confidence, buy_prob, hold_prob, sell_prob = self._probabilities(predicted_return, sigma)

        news_summary = self._weighted_source_summary(news)
        reddit_summary = self._weighted_source_summary(reddit_posts + reddit_comments)
        source_count = len(news) + len(reddit_posts) + len(reddit_comments)
        source_coverage = min(1.0, math.log1p(source_count) / math.log(101.0))
        validation_quality = float(np.clip((validation["directional_accuracy"] - 0.45) / 0.20, 0.0, 1.0))
        evidence_quality = float(np.clip(0.55 * source_coverage + 0.45 * validation_quality, 0.0, 1.0))
        confidence = float(np.clip(base_confidence * (0.60 + 0.40 * evidence_quality), 0.0, 0.95))

        grouped = {"technical": 0.0, "news": 0.0, "reddit": float(reddit_overlay)}
        feature_contrib: dict[str, float] = {}
        for name, value in zip(names, contributions):
            feature_contrib[name] = float(value)
            if name.startswith("news_"):
                grouped["news"] += float(value)
            else:
                grouped["technical"] += float(value)
        if reddit_overlay:
            feature_contrib["reddit_inference_overlay"] = float(reddit_overlay)
        total_abs = sum(abs(v) for v in grouped.values()) or 1.0
        contribution_pct = {k: round(abs(v) / total_abs, 4) for k, v in grouped.items()}
        top_drivers = sorted(feature_contrib.items(), key=lambda pair: abs(pair[1]), reverse=True)[:6]

        current_price = float(history["close"].iloc[-1])
        lower_return = predicted_return - 1.96 * sigma
        upper_return = predicted_return + 1.96 * sigma
        recent_items = sorted(all_items, key=lambda item: (item.published_at, item.engagement), reverse=True)

        return {
            "query": query,
            "company": company.as_dict(),
            "ticker": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "horizon": horizon,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "market": self._trend_metrics(history),
            "sentiment": {
                "news": news_summary,
                "reddit": reddit_summary,
                "combined_score": round(float(np.clip(
                    (0.65 * news_summary["score"] + 0.35 * reddit_summary["score"])
                    if reddit_summary["volume"] else news_summary["score"],
                    -1.0, 1.0
                )), 4),
                "reddit_configured": self.reddit.configured,
                "reddit_scope": "matched public posts plus top comments from the most engaged matched threads; Reddit's API does not provide global comment search",
                "source_errors": source_errors,
            },
            "prediction": {
                "signal": signal,
                "confidence": round(confidence, 4),
                "predicted_return_pct": round(predicted_return * 100.0, 4),
                "predicted_price": round(current_price * (1.0 + predicted_return), 4),
                "lower_bound": round(current_price * (1.0 + lower_return), 4),
                "upper_bound": round(current_price * (1.0 + upper_return), 4),
                "buy_probability": round(buy_prob, 4),
                "hold_probability": round(hold_prob, 4),
                "sell_probability": round(sell_prob, 4),
                "residual_volatility_pct": round(sigma * 100.0, 4),
                "evidence_quality": round(evidence_quality, 4),
                "contribution_mix": contribution_pct,
                "top_drivers": [
                    {"feature": name, "effect": round(value, 6), "direction": "up" if value >= 0 else "down"}
                    for name, value in top_drivers
                ],
            },
            "model": {
                "type": "on_demand_ridge_fusion",
                "feature_names": names,
                "technical_features": list(TECHNICAL_FEATURES),
                "alternative_data_features": ["news_sentiment", "news_volume"],
                "inference_only_features": ["reddit_sentiment", "reddit_volume", "reddit_engagement"],
                "reddit_overlay": {
                    "type": "fixed_bounded_inference_overlay",
                    "max_residual_sigma_fraction": 0.25,
                    "current_effect": round(float(reddit_overlay), 6),
                },
                "training_rows": meta["training_rows"],
                "train_start": meta["train_start"],
                "train_end": meta["train_end"],
                "validation": {
                    "mae_pct_points": round(validation["mae"] * 100.0, 4),
                    "directional_accuracy": round(validation["directional_accuracy"], 4),
                    "correlation": round(validation["correlation"], 4),
                },
                "methodology": (
                    "Ridge regression is fit on historical technical features plus dated news aggregates. "
                    "The latest seven-day Reddit state influences the forecast only through a fixed, bounded inference-time overlay; Reddit content never enters fitted model weights."
                ),
            },
            "evidence": {
                "news_count": len(news),
                "reddit_post_count": len(reddit_posts),
                "reddit_comment_count": len(reddit_comments),
                "items": [item.public_dict() for item in recent_items[:40]],
            },
            "history": [
                {"date": idx.date().isoformat(), "close": round(float(row["close"]), 4)}
                for idx, row in history.tail(260).iterrows()
            ],
            "providers": {
                "market_and_news": "Alpha Vantage documented API",
                "reddit": "Reddit Data API (OAuth)" if self.reddit.configured else "not configured",
                "social_sentiment": "deterministic financial lexicon fallback",
            },
            "disclaimer": (
                "Experimental research output, not investment advice. Confidence measures model/evidence consistency, not the probability of profit."
            ),
        }
