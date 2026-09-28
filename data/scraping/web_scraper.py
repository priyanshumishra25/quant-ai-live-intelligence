"""
Distributed async web scraping pipeline.
Handles news, social media, and financial blogs with proxy rotation,
rate limiting, and CAPTCHA handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncGenerator
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ScrapedArticle:
    url: str
    source: str
    title: str
    body: str
    published_at: datetime | None
    tickers: list[str]
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: str = ""

    def __post_init__(self):
        self.content_hash = hashlib.md5(
            (self.title + self.body).encode()
        ).hexdigest()


@dataclass
class SocialPost:
    platform: str          # reddit | twitter | stocktwits
    post_id: str
    author: str
    text: str
    tickers: list[str]
    upvotes: int = 0
    comments: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    url: str = ""
    subreddit: str | None = None


# ---------------------------------------------------------------------------
# Proxy / User-agent rotation
# ---------------------------------------------------------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


class ProxyManager:
    """Manages a pool of proxies with health tracking."""

    def __init__(self, proxies: list[str] | None = None):
        self._pool: list[str] = proxies or []
        self._failures: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def add_proxy(self, proxy: str) -> None:
        self._pool.append(proxy)

    async def get_proxy(self) -> str | None:
        async with self._lock:
            healthy = [p for p in self._pool if self._failures.get(p, 0) < 3]
            return random.choice(healthy) if healthy else None

    async def report_failure(self, proxy: str) -> None:
        async with self._lock:
            self._failures[proxy] = self._failures.get(proxy, 0) + 1
            logger.warning("Proxy failure #%d: %s", self._failures[proxy], proxy)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket per-domain rate limiter."""

    def __init__(self, calls_per_second: float = 1.0):
        self._interval = 1.0 / calls_per_second
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, domain: str) -> None:
        async with self._lock:
            now = time.monotonic()
            last = self._last.get(domain, 0.0)
            wait = self._interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[domain] = time.monotonic()


# ---------------------------------------------------------------------------
# Ticker extractor
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"\b\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")
_COMMON_WORDS = {
    "I", "A", "THE", "AND", "OR", "BUT", "IN", "ON", "AT", "TO", "FOR",
    "OF", "WITH", "IS", "ARE", "WAS", "BY", "FROM", "US", "UK", "EU", "CEO",
    "IPO", "ETF", "AI", "ML", "API", "Q1", "Q2", "Q3", "Q4", "YTD",
}

def extract_tickers(text: str) -> list[str]:
    tickers: set[str] = set()
    for m in _TICKER_RE.finditer(text):
        t = m.group(1) or m.group(2)
        if t and t not in _COMMON_WORDS and len(t) >= 2:
            tickers.add(t)
    return sorted(tickers)


# ---------------------------------------------------------------------------
# Date parser
# ---------------------------------------------------------------------------

def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Base scraper
# ---------------------------------------------------------------------------

class BaseScraper:
    def __init__(
        self,
        proxy_manager: ProxyManager | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: int = 15,
    ):
        self.proxy_manager = proxy_manager or ProxyManager()
        self.rate_limiter = rate_limiter or RateLimiter(calls_per_second=0.5)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "BaseScraper":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def _get(self, url: str) -> str | None:
        domain = urlparse(url).netloc
        await self.rate_limiter.wait(domain)
        proxy = await self.proxy_manager.get_proxy()
        try:
            assert self._session is not None
            async with self._session.get(
                url, headers=self._headers(), proxy=proxy, ssl=False
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                logger.debug("HTTP %d for %s", resp.status, url)
                return None
        except Exception as exc:
            logger.warning("GET failed %s: %s", url, exc)
            if proxy:
                await self.proxy_manager.report_failure(proxy)
            return None


# ---------------------------------------------------------------------------
# RSS / Feed scraper (Reuters, MarketWatch, CNBC, etc.)
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "marketwatch":     "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "cnbc_finance":    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "yahoo_finance":   "https://finance.yahoo.com/news/rssindex",
    "seeking_alpha":   "https://seekingalpha.com/market_currents.xml",
    "bloomberg_mkts":  "https://feeds.bloomberg.com/markets/news.rss",
}


class RSSFeedScraper(BaseScraper):
    """Lightweight RSS scraper that requires no JS rendering."""

    async def scrape_all(
        self, since: datetime | None = None
    ) -> AsyncGenerator[ScrapedArticle, None]:
        tasks = [self._scrape_feed(name, url) for name, url in RSS_FEEDS.items()]
        for coro in asyncio.as_completed(tasks):
            articles = await coro
            for art in articles:
                if since and art.published_at and art.published_at < since:
                    continue
                yield art

    async def _scrape_feed(self, source: str, url: str) -> list[ScrapedArticle]:
        html = await self._get(url)
        if not html:
            return []
        feed = feedparser.parse(html)
        results = []
        for entry in feed.entries:
            title = entry.get("title", "")
            body  = BeautifulSoup(
                entry.get("summary", entry.get("content", [{"value": ""}])[0].get("value", "")),
                "html.parser"
            ).get_text(" ", strip=True)
            pub = _parse_date(entry.get("published") or entry.get("updated"))
            art = ScrapedArticle(
                url=entry.get("link", url),
                source=source,
                title=title,
                body=body,
                published_at=pub,
                tickers=extract_tickers(title + " " + body),
                author=entry.get("author"),
                tags=[t.get("term", "") for t in entry.get("tags", [])],
            )
            results.append(art)
        logger.debug("%s: fetched %d articles", source, len(results))
        return results


# ---------------------------------------------------------------------------
# Playwright-based scraper (JavaScript-heavy sites)
# ---------------------------------------------------------------------------

class PlaywrightScraper:
    """
    Handles JS-heavy pages: SeekingAlpha full articles,
    StockTwits timeline, etc.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "PlaywrightScraper":
        self._pw = await async_playwright().__aenter__()
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        self._context = await self._browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 800},
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._browser:
            await self._browser.close()
        await self._pw.__aexit__(None, None, None)

    async def get_text(self, url: str, wait_selector: str = "body") -> str | None:
        if not self._context:
            return None
        page = await self._context.new_page()
        try:
            await page.goto(url, timeout=20_000)
            await page.wait_for_selector(wait_selector, timeout=10_000)
            await asyncio.sleep(random.uniform(1.0, 2.5))   # human-like delay
            return await page.content()
        except Exception as exc:
            logger.warning("Playwright error for %s: %s", url, exc)
            return None
        finally:
            await page.close()

    async def scrape_stocktwits(self, ticker: str) -> list[SocialPost]:
        url = f"https://stocktwits.com/symbol/{ticker}"
        html = await self.get_text(url, wait_selector=".StreamMessage")
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        posts = []
        for msg in soup.select(".StreamMessage")[:50]:
            body_el = msg.select_one(".RichTextMessage")
            user_el = msg.select_one(".username")
            if not body_el or not user_el:
                continue
            text = body_el.get_text(" ", strip=True)
            posts.append(SocialPost(
                platform="stocktwits",
                post_id=msg.get("data-message-id", ""),
                author=user_el.get_text(strip=True),
                text=text,
                tickers=[ticker] + extract_tickers(text),
            ))
        return posts


# ---------------------------------------------------------------------------
# Reddit scraper (via Pushshift-style API / PRAW)
# ---------------------------------------------------------------------------

class RedditScraper(BaseScraper):
    """
    Scrapes Reddit via the JSON API (no auth required for public data,
    limited to ~100 recent posts per subreddit).
    """

    SUBREDDITS = [
        "wallstreetbets", "stocks", "investing",
        "options", "cryptocurrency", "StockMarket",
    ]

    async def scrape_subreddit(
        self, subreddit: str, limit: int = 100
    ) -> list[SocialPost]:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
        html = await self._get(url)
        if not html:
            return []
        import json
        try:
            data = json.loads(html)
        except json.JSONDecodeError:
            return []

        posts = []
        for child in data.get("data", {}).get("children", []):
            p = child.get("data", {})
            text = p.get("title", "") + " " + p.get("selftext", "")
            posts.append(SocialPost(
                platform="reddit",
                post_id=p.get("id", ""),
                author=p.get("author", ""),
                text=text.strip(),
                tickers=extract_tickers(text),
                upvotes=p.get("ups", 0),
                comments=p.get("num_comments", 0),
                created_at=datetime.utcfromtimestamp(p.get("created_utc", 0)),
                url="https://reddit.com" + p.get("permalink", ""),
                subreddit=subreddit,
            ))
        return posts

    async def scrape_all(self) -> list[SocialPost]:
        tasks = [self.scrape_subreddit(s) for s in self.SUBREDDITS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_posts: list[SocialPost] = []
        for r in results:
            if isinstance(r, list):
                all_posts.extend(r)
        logger.info("Reddit: collected %d posts", len(all_posts))
        return all_posts


# ---------------------------------------------------------------------------
# News scraper (BeautifulSoup article extraction)
# ---------------------------------------------------------------------------

ARTICLE_SELECTORS: dict[str, dict[str, str]] = {
    "marketwatch.com": {
        "title": "h1.article__headline",
        "body":  "div.article__body",
        "date":  "time[datetime]",
    },
    "finance.yahoo.com": {
        "title": "h1.caas-title-wrapper",
        "body":  "div.caas-body",
        "date":  "time",
    },
    "cnbc.com": {
        "title": "h1.ArticleHeader-headline",
        "body":  "div.ArticleBody-articleBody",
        "date":  "time[datetime]",
    },
}


class ArticleScraper(BaseScraper):
    """Extracts full article text from financial news sites."""

    def _selectors_for(self, url: str) -> dict[str, str]:
        domain = urlparse(url).netloc.lstrip("www.")
        return ARTICLE_SELECTORS.get(
            domain,
            {"title": "h1", "body": "article", "date": "time"},
        )

    async def scrape_article(self, url: str, source: str) -> ScrapedArticle | None:
        html = await self._get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        sel = self._selectors_for(url)

        title_el = soup.select_one(sel["title"])
        body_el   = soup.select_one(sel["body"])
        date_el   = soup.select_one(sel["date"])

        if not title_el or not body_el:
            return None

        title = title_el.get_text(" ", strip=True)
        body  = body_el.get_text(" ", strip=True)
        pub   = _parse_date(date_el.get("datetime") if date_el else None)

        return ScrapedArticle(
            url=url,
            source=source,
            title=title,
            body=body[:5000],       # cap at 5k chars
            published_at=pub,
            tickers=extract_tickers(title + " " + body),
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ScrapingOrchestrator:
    """
    Coordinates all scrapers and deduplicates results.
    Runs continuously, sleeping between cycles.
    """

    def __init__(self, cycle_minutes: int = 30):
        self.cycle_minutes = cycle_minutes
        self._seen_hashes: set[str] = set()

    def _dedup(self, articles: list[ScrapedArticle]) -> list[ScrapedArticle]:
        fresh = []
        for art in articles:
            if art.content_hash not in self._seen_hashes:
                self._seen_hashes.add(art.content_hash)
                fresh.append(art)
        return fresh

    async def run_once(self) -> tuple[list[ScrapedArticle], list[SocialPost]]:
        since = datetime.utcnow() - timedelta(hours=2)
        articles: list[ScrapedArticle] = []
        social:   list[SocialPost]     = []

        # RSS feeds
        async with RSSFeedScraper() as rss:
            async for art in rss.scrape_all(since=since):
                articles.append(art)

        # Reddit
        async with RedditScraper() as reddit:
            social.extend(await reddit.scrape_all())

        articles = self._dedup(articles)
        logger.info(
            "Scraping cycle done: %d articles, %d social posts",
            len(articles), len(social),
        )
        return articles, social

    async def run_forever(self) -> AsyncGenerator[
        tuple[list[ScrapedArticle], list[SocialPost]], None
    ]:
        while True:
            try:
                yield await self.run_once()
            except Exception as exc:
                logger.error("Scraping cycle error: %s", exc)
            await asyncio.sleep(self.cycle_minutes * 60)
