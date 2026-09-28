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
    assert payload["model"]["validation"]["purge_rows"] == 5
    assert payload["model"]["validation"]["rows"] >= 10
    assert payload["model"]["validation"]["effective_non_overlapping_observations"] >= 1
    assert payload["model"]["validation"]["reliability"] in {"LOW", "MEDIUM", "HIGH"}
    assert payload["model"]["ridge_alpha_selection"]["method"] == "training_only_generalized_cross_validation"
    assert "news_sentiment" in payload["model"]["feature_names"]
    assert "reddit_sentiment" not in payload["model"]["feature_names"]
    assert "reddit_sentiment" in payload["model"]["inference_only_features"]
    assert payload["model"]["reddit_overlay"]["type"] == "fixed_bounded_inference_overlay"
    assert 0.0 <= payload["prediction"]["signal_strength"] <= 1.0
    assert -1.0 <= payload["prediction"]["signal_score"] <= 1.0
    assert abs(
        payload["prediction"]["bullish_score"]
        + payload["prediction"]["neutral_score"]
        + payload["prediction"]["bearish_score"]
        - 1.0
    ) < 0.001
    assert "buy_probability" not in payload["prediction"]
    assert "confidence" not in payload["prediction"]
    assert payload["prediction"]["holdout_residual_volatility_pct"] == pytest.approx(
        payload["model"]["validation"]["residual_std_pct_points"]
    )
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

@pytest.mark.asyncio
async def test_provider_level_cache_prevents_repeat_alpha_calls_across_horizons():
    from services.cache import MemoryCache

    class CountingAlpha(FakeAlpha):
        outputsize = "compact"

        def __init__(self):
            self.search_calls = 0
            self.history_calls = 0
            self.news_calls = 0

        async def search(self, query: str, limit: int = 8):
            self.search_calls += 1
            return await super().search(query, limit)

        async def daily_history(self, symbol: str, max_rows: int = 1500):
            self.history_calls += 1
            return await super().daily_history(symbol, max_rows)

        async def news(self, symbol: str, lookback_days: int = 365, limit: int = 500):
            self.news_calls += 1
            return await super().news(symbol, lookback_days, limit)

    alpha = CountingAlpha()
    service = LiveIntelligenceService(
        alpha,
        FakeReddit(),
        cache=MemoryCache(),
        news_fetch_limit=100,
        reddit_fetch_limit=20,
    )
    await service.analyze("Acme Robotics", "5d", include_reddit_comments=False)
    await service.analyze("Acme Robotics", "20d", include_reddit_comments=False)

    assert alpha.search_calls == 1
    assert alpha.history_calls == 1
    assert alpha.news_calls == 1


@pytest.mark.asyncio
async def test_explicit_uppercase_ticker_skips_symbol_search():
    from services.cache import MemoryCache

    class CountingAlpha(FakeAlpha):
        outputsize = "compact"

        def __init__(self):
            self.search_calls = 0

        async def search(self, query: str, limit: int = 8):
            self.search_calls += 1
            return await super().search(query, limit)

    alpha = CountingAlpha()
    service = LiveIntelligenceService(alpha, FakeReddit(), cache=MemoryCache())
    payload = await service.analyze("ACME", "5d", include_reddit_comments=False)
    assert payload["ticker"] == "ACME"
    assert alpha.search_calls == 0


@pytest.mark.asyncio
async def test_alpha_free_tier_defaults_to_compact_and_translates_quota_message():
    import httpx
    from services.live_intelligence import AlphaVantageClient, ProviderLimitError

    seen_outputsize: list[str] = []

    def daily_handler(request: httpx.Request):
        params = dict(request.url.params)
        seen_outputsize.append(params.get("outputsize", ""))
        return httpx.Response(200, json={"Time Series (Daily)": {
            "2026-09-25": {"1. open":"100","2. high":"101","3. low":"99","4. close":"100.5","5. volume":"1000"}
        }})

    async with httpx.AsyncClient(transport=httpx.MockTransport(daily_handler)) as client:
        alpha = AlphaVantageClient("test", client=client, min_interval_seconds=0)
        await alpha.daily_history("AAPL")
    assert seen_outputsize == ["compact"]

    message = (
        "Thank you for using Alpha Vantage! Please consider spreading out your free API requests more sparingly "
        "(1 request per second). The free key rate limit is 25 requests per day."
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"Information": message}))) as client:
        alpha = AlphaVantageClient("test", client=client, min_interval_seconds=0)
        with pytest.raises(ProviderLimitError) as exc:
            await alpha.search("Apple")
    assert exc.value.kind == "daily_quota"
    assert "daily quota reached" in str(exc.value).lower()


def test_news_mapping_respects_us_market_close():
    dates = pd.DatetimeIndex([pd.Timestamp("2026-09-18"), pd.Timestamp("2026-09-21")])
    before_close = IntelligenceItem(
        source="Wire", kind="news", title="before", text="before",
        published_at=datetime(2026, 9, 18, 19, 30, tzinfo=timezone.utc),  # 15:30 ET
        url="https://example.com/before", sentiment=0.8, relevance=1.0,
    )
    after_close = IntelligenceItem(
        source="Wire", kind="news", title="after", text="after",
        published_at=datetime(2026, 9, 18, 20, 30, tzinfo=timezone.utc),  # 16:30 ET
        url="https://example.com/after", sentiment=-0.8, relevance=1.0,
    )
    first = LiveIntelligenceService._map_items_to_market_dates(dates, [before_close])
    second = LiveIntelligenceService._map_items_to_market_dates(dates, [after_close])
    assert first.loc[pd.Timestamp("2026-09-18"), "news_volume"] > 0
    assert second.loc[pd.Timestamp("2026-09-18"), "news_volume"] == 0
    assert second.loc[pd.Timestamp("2026-09-21"), "news_volume"] > 0


@pytest.mark.asyncio
async def test_alpha_drops_news_with_invalid_timestamp():
    import httpx
    from services.live_intelligence import AlphaVantageClient

    payload = {"feed": [
        {
            "title": "bad timestamp", "summary": "ignored", "time_published": "not-a-time",
            "source": "Wire", "url": "https://example.com/bad",
            "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "1", "ticker_sentiment_score": "0.2"}],
        },
        {
            "title": "valid timestamp", "summary": "kept", "time_published": "20260920T120000",
            "source": "Wire", "url": "https://example.com/good",
            "ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "1", "ticker_sentiment_score": "0.3"}],
        },
    ]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
        alpha = AlphaVantageClient("test", client=client, min_interval_seconds=0)
        items = await alpha.news("AAPL")
    assert [item.title for item in items] == ["valid timestamp"]


def test_twenty_day_validation_is_purged_and_flagged_low_reliability_on_compact_history():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2026-05-01", periods=100)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates))))
    history = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(len(dates), 1_000_000),
    }, index=dates)
    _, validation, meta = LiveIntelligenceService._fit_fusion_model(history, [], "20d")
    assert validation["purge_rows"] == 20
    assert validation["rows"] >= 10
    assert validation["effective_non_overlapping_observations"] <= 2
    assert validation["reliability"] == "LOW"
    assert pd.Timestamp(meta["train_end"]) < pd.Timestamp(meta["validation_start"])
