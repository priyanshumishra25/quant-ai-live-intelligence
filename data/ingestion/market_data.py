"""
data/ingestion/market_data.py
Async market data ingestion from multiple providers with fallback logic,
rate limiting, caching, and corporate action adjustment.
"""

from __future__ import annotations

import asyncio
import io
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
import redis.asyncio as aioredis
import yfinance as yf
from loguru import logger

from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Base downloader interface
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataDownloader:
    """Unified interface for all market data providers."""

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError

    async def fetch_realtime(self, tickers: list[str]) -> dict[str, float]:
        raise NotImplementedError

    async def fetch_options_chain(self, ticker: str) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance (free, rate-limited, good for daily data)
# ─────────────────────────────────────────────────────────────────────────────

class YahooFinanceDownloader(MarketDataDownloader):
    """
    Uses yfinance.  Good for 1m–1d historical.  Auto-adjusts for splits/dividends.
    Thread-pool backed so it doesn't block the event loop.
    """

    _INTERVAL_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "60m", "4h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
    }

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            lambda: self._download_sync(ticker, start, end, interval),
        )
        return df

    def _download_sync(
        self, ticker: str, start: date, end: date, interval: str
    ) -> pd.DataFrame:
        yf_interval = self._INTERVAL_MAP.get(interval, "1d")
        t = yf.Ticker(ticker)
        df = t.history(
            start=str(start),
            end=str(end + timedelta(days=1)),  # end is exclusive in yfinance
            interval=yf_interval,
            auto_adjust=True,
            prepost=False,
        )
        if df.empty:
            logger.warning(f"No data from Yahoo Finance for {ticker}")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
        df.index.name = "datetime"
        df.columns = df.columns.str.lower()
        df["ticker"] = ticker
        return df[["open", "high", "low", "close", "volume", "ticker"]]

    async def fetch_realtime(self, tickers: list[str]) -> dict[str, float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._realtime_sync, tickers)

    def _realtime_sync(self, tickers: list[str]) -> dict[str, float]:
        data = {}
        for tk in tickers:
            try:
                info = yf.Ticker(tk).fast_info
                data[tk] = float(info.last_price or 0)
            except Exception as e:
                logger.error(f"Yahoo realtime failed for {tk}: {e}")
                data[tk] = 0.0
        return data

    async def fetch_options_chain(self, ticker: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._options_sync, ticker)

    def _options_sync(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        chain = {}
        for exp in (t.options or []):
            try:
                c, p = t.option_chain(exp)
                chain[exp] = {"calls": c.to_dict("records"), "puts": p.to_dict("records")}
            except Exception as e:
                logger.error(f"Options chain error {ticker} {exp}: {e}")
        return chain


# ─────────────────────────────────────────────────────────────────────────────
# Polygon.io (paid, tick-level, professional grade)
# ─────────────────────────────────────────────────────────────────────────────

class PolygonDownloader(MarketDataDownloader):
    """
    Polygon.io REST + WebSocket.  Supports tick-level, options, forex, crypto.
    Requires paid subscription for real-time.
    """

    BASE = "https://api.polygon.io"

    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.data.polygon_key}"},
            timeout=30.0,
            limits=httpx.Limits(max_connections=20),
        )

    async def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        multiplier, timespan = self._parse_interval(interval)
        url = (
            f"{self.BASE}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{start}/{end}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        rows = []
        while url:
            r = await self._client.get(url, params=params)
            r.raise_for_status()
            body = r.json()
            rows.extend(body.get("results", []))
            url = body.get("next_url")
            params = {}   # next_url already contains all params

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["ticker"] = ticker
        df = df.set_index("datetime")[["open", "high", "low", "close", "volume", "ticker"]]
        return df

    def _parse_interval(self, interval: str) -> tuple[int, str]:
        mapping = {
            "1m": (1, "minute"), "5m": (5, "minute"), "15m": (15, "minute"),
            "30m": (30, "minute"), "1h": (1, "hour"), "4h": (4, "hour"),
            "1d": (1, "day"), "1wk": (1, "week"), "1mo": (1, "month"),
        }
        return mapping.get(interval, (1, "day"))

    async def fetch_realtime(self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        tasks = [self._fetch_last_price(tk) for tk in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for tk, res in zip(tickers, results):
            prices[tk] = float(res) if isinstance(res, (int, float)) else 0.0
        return prices

    async def _fetch_last_price(self, ticker: str) -> float:
        r = await self._client.get(f"{self.BASE}/v2/last/trade/{ticker}")
        r.raise_for_status()
        return r.json()["results"]["p"]

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-provider downloader with fallback
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataOrchestrator:
    """
    Tries providers in priority order.  On failure, falls back to the next.
    Caches results in Redis to minimise external calls.
    """

    def __init__(self):
        self._primary = PolygonDownloader()
        self._fallback = YahooFinanceDownloader()

    async def get_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Return OHLCV dataframe.  Attempts Polygon first, Yahoo on error.
        """
        cache_key = f"ohlcv:{ticker}:{start}:{end}:{interval}"

        if use_cache:
            cached = await _redis_get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return pd.read_parquet(io.BytesIO(cached))

        # Try primary provider
        df = pd.DataFrame()
        try:
            if settings.data.polygon_key:
                df = await self._primary.fetch_ohlcv(ticker, start, end, interval)
        except Exception as e:
            logger.warning(f"Polygon failed for {ticker}: {e}, falling back to Yahoo")

        # Fallback
        if df.empty:
            try:
                df = await self._fallback.fetch_ohlcv(ticker, start, end, interval)
            except Exception as e:
                logger.error(f"All providers failed for {ticker}: {e}")
                return pd.DataFrame()

        if not df.empty and use_cache:
            ttl = 60 if interval in ("1m", "5m") else 3600
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=True)
            await _redis_set(cache_key, buffer.getvalue(), ttl=ttl)

        return df

    async def get_bulk_ohlcv(
        self,
        tickers: list[str],
        start: date,
        end: date,
        interval: str = "1d",
        max_concurrent: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple tickers concurrently with a semaphore."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _fetch(tk: str) -> tuple[str, pd.DataFrame]:
            async with sem:
                df = await self.get_ohlcv(tk, start, end, interval)
                return tk, df

        results = await asyncio.gather(*[_fetch(tk) for tk in tickers])
        return dict(results)

    async def get_realtime(self, tickers: list[str]) -> dict[str, float]:
        try:
            if settings.data.polygon_key:
                return await self._primary.fetch_realtime(tickers)
        except Exception as e:
            logger.warning(f"Polygon realtime failed: {e}")
        return await self._fallback.fetch_realtime(tickers)

    async def get_options_chain(self, ticker: str) -> dict:
        return await self._fallback.fetch_options_chain(ticker)


# ─────────────────────────────────────────────────────────────────────────────
# SEC Filings (EDGAR)
# ─────────────────────────────────────────────────────────────────────────────

class SECFilingsDownloader:
    """
    Downloads 10-K, 10-Q, 8-K filings from the SEC EDGAR API.
    Useful for fundamental data and earnings transcripts.
    """

    BASE = "https://data.sec.gov"
    HEADERS = {"User-Agent": "QuantAI research@quantai.com"}

    async def get_company_facts(self, cik: str) -> dict:
        """Return all XBRL facts for a company (revenue, EPS, etc.)."""
        async with httpx.AsyncClient(headers=self.HEADERS) as client:
            r = await client.get(f"{self.BASE}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json")
            r.raise_for_status()
            return r.json()

    async def get_recent_filings(self, cik: str, form_type: str = "10-K") -> list[dict]:
        """Return metadata for recent filings of a given type."""
        async with httpx.AsyncClient(headers=self.HEADERS) as client:
            r = await client.get(f"{self.BASE}/submissions/CIK{cik.zfill(10)}.json")
            r.raise_for_status()
            body = r.json()
            recent = body.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            return [
                {"accession": a, "date": d}
                for f, a, d in zip(forms, accessions, dates)
                if f == form_type
            ][:20]


# ─────────────────────────────────────────────────────────────────────────────
# Macro data (FRED via pandas-datareader)
# ─────────────────────────────────────────────────────────────────────────────

MACRO_SERIES = {
    "fed_rate":     "FEDFUNDS",
    "cpi":          "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp_growth":   "A191RL1Q225SBEA",
    "ten_yr_yield": "DGS10",
    "two_yr_yield": "DGS2",
    "vix":          "VIXCLS",
    "dxy":          "DTWEXBGS",
    "oil_wti":      "DCOILWTICO",
    "gold":         "GOLDAMGBD228NLBM",
    "m2":           "M2SL",
}


async def fetch_macro_data(
    start: date = date(2000, 1, 1),
    end: date | None = None,
) -> pd.DataFrame:
    """
    Fetch macroeconomic indicators from FRED.
    Returns a combined DataFrame indexed by date.
    """
    import pandas_datareader as pdr

    end = end or date.today()
    loop = asyncio.get_event_loop()

    def _fetch():
        frames = {}
        for name, series_id in MACRO_SERIES.items():
            try:
                s = pdr.get_data_fred(series_id, start=str(start), end=str(end))
                frames[name] = s.iloc[:, 0]
            except Exception as e:
                logger.warning(f"FRED series {series_id} failed: {e}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index)
        # Forward-fill monthly/quarterly series to daily
        df = df.asfreq("D").ffill()
        return df

    return await loop.run_in_executor(None, _fetch)


# ─────────────────────────────────────────────────────────────────────────────
# Redis cache helpers
# ─────────────────────────────────────────────────────────────────────────────

_redis_client: aioredis.Redis | None = None


def _get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.infra.redis_url, decode_responses=False)
    return _redis_client


async def _redis_get(key: str) -> bytes | None:
    """Read a binary cache entry; cache failures never block market-data fetches."""
    try:
        return await _get_redis_client().get(key)
    except Exception as exc:
        logger.debug(f"Redis read skipped for {key}: {exc}")
        return None


async def _redis_set(key: str, value: bytes, ttl: int = 3600) -> None:
    """Write a binary cache entry with TTL; failures are non-fatal."""
    try:
        await _get_redis_client().setex(key, ttl, value)
    except Exception as exc:
        logger.debug(f"Redis write skipped for {key}: {exc}")
