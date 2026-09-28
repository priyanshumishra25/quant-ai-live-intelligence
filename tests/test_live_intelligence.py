from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from services.live_intelligence import (
    FinancialLexiconSentiment,
    IntelligenceItem,
    LiveIntelligenceService,
    StockCandidate,
)


class FakeAlpha:
    configured = True

    async def close(self):
        return None

    async def search(self, query: str, limit: int = 8):
        return [StockCandidate(symbol="ACME", name="Acme Robotics", match_score=0.99)][:limit]

    async def resolve(self, query: str):
        return StockCandidate(symbol="ACME", name="Acme Robotics", region="United States", currency="USD", match_score=0.99)

    async def daily_history(self, symbol: str, max_rows: int = 1500):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2023-01-02", periods=520)
        ret = 0.0005 + rng.normal(0, 0.012, len(dates))
        close = 80 * np.exp(np.cumsum(ret))
        return pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.002, len(dates))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
        }, index=dates)

    async def news(self, symbol: str, lookback_days: int = 365, limit: int = 500):
        now = datetime.now(timezone.utc)
        return [
            IntelligenceItem(
                source="Example News",
                kind="news",
                title=f"Acme growth update {i}",
                text="Strong demand and profit growth beat expectations",
                published_at=now - timedelta(days=i * 3),
                url=f"https://example.com/news/{i}",
                sentiment=0.55 if i % 3 else -0.2,
                relevance=0.9,
            )
            for i in range(100)
        ]


class FakeReddit:
    configured = True

    async def close(self):
        return None

    async def search_posts(self, ticker: str, company_name: str, limit: int = 100):
        now = datetime.now(timezone.utc)
        return [
            IntelligenceItem(
                source="Reddit",
                kind="reddit_post",
                title=f"ACME discussion {i}",
                text="bullish growth" if i % 2 else "risk of weak demand",
                published_at=now - timedelta(days=i * 4),
                url=f"https://www.reddit.com/r/stocks/comments/{i}",
                sentiment=0.45 if i % 2 else -0.35,
                engagement=10 + i,
                subreddit="stocks",
                metadata={"id": str(i)},
            )
            for i in range(min(limit, 70))
        ]

    async def top_comments(self, posts, max_posts: int = 3, comments_per_post: int = 15):
        now = datetime.now(timezone.utc)
        return [
            IntelligenceItem(
                source="Reddit",
                kind="reddit_comment",
                title="Comment in ACME thread",
                text="strong product but valuation risk",
                published_at=now - timedelta(hours=i + 1),
                url=f"https://www.reddit.com/r/stocks/comments/c{i}",
                sentiment=0.1,
                engagement=5 + i,
                subreddit="stocks",
            )
            for i in range(8)
        ]


def test_financial_lexicon_has_direction_and_negation():
    scorer = FinancialLexiconSentiment()
    assert scorer.score("strong growth beats expectations, bullish demand") > 0.3
    assert scorer.score("weak demand, downgrade, losses and fraud probe") < -0.3
    assert scorer.score("not weak, no loss") > -0.5


@pytest.mark.asyncio
async def test_live_intelligence_fuses_history_news_and_reddit():
    service = LiveIntelligenceService(FakeAlpha(), FakeReddit(), lookback_days=365, news_fetch_limit=100, reddit_fetch_limit=70)
    payload = await service.analyze("Acme Robotics", "5d", include_reddit_comments=True)

    assert payload["ticker"] == "ACME"
    assert payload["company"]["name"] == "Acme Robotics"
    assert payload["market"]["history_rows"] == 520
    assert payload["evidence"]["news_count"] == 100
    assert payload["evidence"]["reddit_post_count"] == 70
    assert payload["evidence"]["reddit_comment_count"] == 8
    assert payload["model"]["type"] == "on_demand_ridge_fusion"
    assert payload["model"]["training_rows"] > 100
    assert "news_sentiment" in payload["model"]["feature_names"]
    assert "reddit_sentiment" not in payload["model"]["feature_names"]
    assert "reddit_sentiment" in payload["model"]["inference_only_features"]
    assert payload["model"]["reddit_overlay"]["type"] == "fixed_bounded_inference_overlay"
    assert 0.0 <= payload["prediction"]["confidence"] <= 0.95
    assert abs(
        payload["prediction"]["buy_probability"]
        + payload["prediction"]["hold_probability"]
        + payload["prediction"]["sell_probability"]
        - 1.0
    ) < 0.001
    assert set(payload["prediction"]["contribution_mix"]) == {"technical", "news", "reddit"}
    assert payload["history"]
    assert payload["evidence"]["items"]


@pytest.mark.asyncio
async def test_search_delegates_to_provider_resolution():
    service = LiveIntelligenceService(FakeAlpha(), FakeReddit())
    matches = await service.search("Acme")
    assert matches[0]["symbol"] == "ACME"
    assert matches[0]["name"] == "Acme Robotics"

@pytest.mark.asyncio
async def test_alpha_vantage_adapter_parses_documented_payloads():
    import httpx
    from services.live_intelligence import AlphaVantageClient

    dates = pd.bdate_range("2025-01-02", periods=80)
    daily = {
        d.date().isoformat(): {
            "1. open": "100", "2. high": "102", "3. low": "99", "4. close": str(100 + i * 0.2), "5. volume": "1234567"
        }
        for i, d in enumerate(dates)
    }

    def handler(request: httpx.Request):
        params = dict(request.url.params)
        fn = params.get("function")
        if fn == "SYMBOL_SEARCH":
            return httpx.Response(200, json={"bestMatches": [{
                "1. symbol": "AAPL", "2. name": "Apple Inc", "3. type": "Equity",
                "4. region": "United States", "5. marketOpen": "09:30", "6. marketClose": "16:00",
                "7. timezone": "UTC-04", "8. currency": "USD", "9. matchScore": "0.999",
            }]})
        if fn == "TIME_SERIES_DAILY":
            return httpx.Response(200, json={"Time Series (Daily)": daily})
        if fn == "NEWS_SENTIMENT":
            return httpx.Response(200, json={"feed": [{
                "title": "Apple demand remains strong", "summary": "Growth beats expectations",
                "time_published": "20260920T120000", "source": "Example Wire", "url": "https://example.com/apple",
                "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "0.95", "ticker_sentiment_score": "0.42"}],
            }]})
        return httpx.Response(400, json={"Error Message": "unexpected"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        alpha = AlphaVantageClient("test-key", client=client)
        match = await alpha.resolve("Apple")
        history = await alpha.daily_history("AAPL")
        news = await alpha.news("AAPL")
    assert match.symbol == "AAPL"
    assert len(history) == 80
    assert history.index.is_monotonic_increasing
    assert news[0].sentiment == pytest.approx(0.42)
    assert news[0].relevance == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_reddit_adapter_uses_oauth_and_parses_posts_and_comments():
    import httpx
    from services.live_intelligence import RedditClient

    def handler(request: httpx.Request):
        if request.url.host == "www.reddit.com" and request.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        if request.url.path == "/search":
            return httpx.Response(200, json={"data": {"after": None, "children": [{"kind": "t3", "data": {
                "id": "abc123", "title": "AAPL looks strong", "selftext": "bullish demand and growth",
                "created_utc": 1_790_000_000, "score": 25, "num_comments": 7,
                "permalink": "/r/stocks/comments/abc123/aapl/", "subreddit": "stocks",
            }}]}})
        if request.url.path == "/comments/abc123":
            return httpx.Response(200, json=[{"data": {"children": []}}, {"data": {"children": [{"kind": "t1", "data": {
                "body": "strong product, but valuation risk", "created_utc": 1_790_000_100, "score": 8,
                "permalink": "/r/stocks/comments/abc123/aapl/comment1/",
            }}]}}])
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        reddit = RedditClient("client-id", "secret", "QuantAIResearch/3.0 test", client=client)
        posts = await reddit.search_posts("AAPL", "Apple Inc", limit=10)
        comments = await reddit.top_comments(posts, max_posts=1, comments_per_post=10)
    assert len(posts) == 1
    assert posts[0].subreddit == "stocks"
    assert posts[0].engagement == 32
    assert posts[0].sentiment > 0
    assert len(comments) == 1
    assert comments[0].url.startswith("https://www.reddit.com/")
