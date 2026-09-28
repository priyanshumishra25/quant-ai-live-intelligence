"""
features/nlp/sentiment_pipeline.py
Full NLP pipeline: news scraping → cleaning → FinBERT sentiment → 
fear/greed scoring → aggregated market signals.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import praw
import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentimentRecord:
    text: str
    source: str          # "news" | "reddit" | "stocktwits" | "twitter"
    ticker: str
    timestamp: datetime
    raw_score: float     # -1.0 (bearish) to +1.0 (bullish)
    label: str           # "positive" | "negative" | "neutral"
    confidence: float
    emotion: str = ""    # "fear" | "greed" | "uncertainty" | "euphoria"
    engagement: int = 0  # upvotes, likes, retweets
    url: str = ""


@dataclass
class AggregatedSentiment:
    ticker: str
    timestamp: datetime
    bullish_score: float        # -1 to +1
    fear_greed_index: float     # 0=extreme fear, 100=extreme greed
    sentiment_momentum: float   # rate of change in sentiment
    volume: int                 # number of mentions
    engagement: int             # total social engagement
    top_keywords: list[str] = field(default_factory=list)
    sources: dict[str, float] = field(default_factory=dict)  # per-source scores


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

_FINBERT_MODEL = None
_ROBERTA_MODEL = None


def _get_finbert():
    """Load FinBERT (ProsusAI/finbert) — tuned on financial text."""
    global _FINBERT_MODEL
    if _FINBERT_MODEL is None:
        model_name = "ProsusAI/finbert"
        logger.info(f"Loading {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        _FINBERT_MODEL = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            max_length=512,
            truncation=True,
        )
    return _FINBERT_MODEL


def _get_roberta_emotions():
    """Load RoBERTa fine-tuned on emotion detection (SamLowe/roberta-base-go_emotions)."""
    global _ROBERTA_MODEL
    if _ROBERTA_MODEL is None:
        model_name = "SamLowe/roberta-base-go_emotions"
        logger.info(f"Loading {model_name}...")
        device = 0 if torch.cuda.is_available() else -1
        _ROBERTA_MODEL = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            max_length=512,
            truncation=True,
            top_k=3,
        )
    return _ROBERTA_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise raw social/news text for the NLP pipeline."""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)          # keep hashtag words
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()
    return text[:1024]   # truncate very long texts


def extract_tickers(text: str) -> list[str]:
    """Extract $TICKER mentions from text."""
    return list(set(_TICKER_RE.findall(text.upper())))


# ─────────────────────────────────────────────────────────────────────────────
# FinBERT Sentiment Scoring
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_SCORE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def score_sentiment_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Run FinBERT on a batch of texts.
    Returns list of {label, score, raw_score} dicts.
    """
    clf = _get_finbert()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = clf(batch, truncation=True, max_length=512)
        for pred in preds:
            label = pred["label"].lower()
            confidence = pred["score"]
            raw = _LABEL_SCORE.get(label, 0.0) * confidence
            results.append({
                "label": label,
                "confidence": confidence,
                "raw_score": raw,
            })
    return results


def detect_emotions_batch(texts: list[str], batch_size: int = 32) -> list[str]:
    """
    Classify text into financial emotions: fear, greed, uncertainty, euphoria, neutral.
    Uses go_emotions model and remaps to financial emotion categories.
    """
    GREED_EMOTIONS = {"excitement", "joy", "optimism", "love", "admiration"}
    FEAR_EMOTIONS = {"fear", "nervousness", "anxiety", "sadness", "disappointment"}
    UNCERTAINTY_EMOTIONS = {"confusion", "surprise", "curiosity", "realization"}
    EUPHORIA_EMOTIONS = {"amusement", "pride", "relief", "gratitude"}

    clf = _get_roberta_emotions()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = clf(batch, truncation=True, max_length=512)
        for pred_list in preds:
            top = pred_list[0]["label"] if pred_list else "neutral"
            if top in GREED_EMOTIONS:
                results.append("greed")
            elif top in FEAR_EMOTIONS:
                results.append("fear")
            elif top in EUPHORIA_EMOTIONS:
                results.append("euphoria")
            elif top in UNCERTAINTY_EMOTIONS:
                results.append("uncertainty")
            else:
                results.append("neutral")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reddit Collector
# ─────────────────────────────────────────────────────────────────────────────

SUBREDDITS = [
    "wallstreetbets", "stocks", "investing",
    "options", "stockmarket", "SecurityAnalysis",
]


class RedditCollector:
    """Collects posts and comments from investing subreddits using PRAW."""

    def __init__(self):
        self._reddit = praw.Reddit(
            client_id=settings.data.reddit_client_id,
            client_secret=settings.data.reddit_secret,
            user_agent=settings.data.reddit_user_agent,
            read_only=True,
        )

    async def collect_hot(
        self,
        tickers: list[str],
        subreddits: list[str] = SUBREDDITS,
        limit: int = 100,
    ) -> list[SentimentRecord]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._collect_sync, tickers, subreddits, limit
        )

    def _collect_sync(
        self,
        tickers: list[str],
        subreddits: list[str],
        limit: int,
    ) -> list[SentimentRecord]:
        ticker_set = set(tickers)
        records = []

        for sub_name in subreddits:
            try:
                sub = self._reddit.subreddit(sub_name)
                for post in sub.hot(limit=limit):
                    mentioned = extract_tickers(post.title + " " + (post.selftext or ""))
                    relevant = mentioned & ticker_set or (not tickers and mentioned)
                    if not relevant:
                        continue

                    text = clean_text(f"{post.title}. {post.selftext or ''}")
                    if not text:
                        continue

                    for ticker in (relevant if relevant else mentioned):
                        records.append(SentimentRecord(
                            text=text[:512],
                            source="reddit",
                            ticker=ticker,
                            timestamp=datetime.utcfromtimestamp(post.created_utc),
                            raw_score=0.0,      # filled in scoring step
                            label="neutral",
                            confidence=0.0,
                            engagement=post.score + post.num_comments,
                            url=f"https://reddit.com{post.permalink}",
                        ))
            except Exception as e:
                logger.error(f"Reddit error for r/{sub_name}: {e}")

        return records


# ─────────────────────────────────────────────────────────────────────────────
# News Collector (NewsAPI + RSS scraping)
# ─────────────────────────────────────────────────────────────────────────────

class NewsCollector:
    """Fetches headlines from NewsAPI and scrapes major financial RSS feeds."""

    NEWS_FEEDS = {
        "reuters": "https://feeds.reuters.com/reuters/businessNews",
        "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "marketwatch": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines",
        "seekingalpha": "https://seekingalpha.com/feed.xml",
    }

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=15.0)

    async def fetch_newsapi(
        self,
        tickers: list[str],
        days_back: int = 3,
    ) -> list[SentimentRecord]:
        if not settings.data.newsapi_key:
            return []

        records = []
        from_dt = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        for ticker in tickers:
            try:
                r = await self._client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": ticker,
                        "from": from_dt,
                        "language": "en",
                        "sortBy": "relevancy",
                        "pageSize": 50,
                        "apiKey": settings.data.newsapi_key,
                    },
                )
                r.raise_for_status()
                for article in r.json().get("articles", []):
                    text = clean_text(
                        f"{article.get('title','')}. {article.get('description','')}"
                    )
                    if not text:
                        continue
                    records.append(SentimentRecord(
                        text=text,
                        source="news",
                        ticker=ticker,
                        timestamp=datetime.fromisoformat(
                            article.get("publishedAt", "").rstrip("Z")
                        ) if article.get("publishedAt") else datetime.utcnow(),
                        raw_score=0.0,
                        label="neutral",
                        confidence=0.0,
                        url=article.get("url", ""),
                    ))
            except Exception as e:
                logger.error(f"NewsAPI error for {ticker}: {e}")

        return records

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# Main Sentiment Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SentimentPipeline:
    """
    Orchestrates collection → cleaning → scoring → aggregation.
    Produces AggregatedSentiment objects per ticker.
    """

    def __init__(self):
        self._reddit = RedditCollector()
        self._news = NewsCollector()

    async def run(
        self,
        tickers: list[str],
        days_back: int = 1,
    ) -> dict[str, AggregatedSentiment]:
        """Full pipeline: collect → score → aggregate → return per-ticker signal."""

        # 1. Collect
        logger.info(f"Collecting sentiment for {tickers}")
        reddit_records, news_records = await asyncio.gather(
            self._reddit.collect_hot(tickers),
            self._news.fetch_newsapi(tickers, days_back=days_back),
        )
        all_records = reddit_records + news_records
        if not all_records:
            logger.warning("No sentiment records collected")
            return {}

        # 2. Score (batch GPU inference)
        texts = [r.text for r in all_records]
        scores = score_sentiment_batch(texts)
        emotions = detect_emotions_batch(texts)

        for rec, s, e in zip(all_records, scores, emotions):
            rec.raw_score = s["raw_score"]
            rec.label = s["label"]
            rec.confidence = s["confidence"]
            rec.emotion = e

        # 3. Aggregate per ticker
        df = pd.DataFrame([
            {
                "ticker": r.ticker,
                "raw_score": r.raw_score,
                "confidence": r.confidence,
                "emotion": r.emotion,
                "engagement": r.engagement,
                "source": r.source,
                "timestamp": r.timestamp,
            }
            for r in all_records
        ])

        aggregated = {}
        for ticker, grp in df.groupby("ticker"):
            agg = self._aggregate(ticker, grp)
            aggregated[ticker] = agg

        return aggregated

    def _aggregate(self, ticker: str, df: pd.DataFrame) -> AggregatedSentiment:
        """
        Weighted aggregation:
        - Engagement-weighted sentiment score
        - Fear/Greed Index (0-100 scale)
        - Sentiment momentum (recent vs older)
        """
        # Engagement-weighted average sentiment
        weights = (df["engagement"] + 1).values  # +1 to avoid zero weights
        bullish = np.average(df["raw_score"].values, weights=weights)

        # Fear/Greed: map -1..+1 to 0..100
        fear_greed = (bullish + 1) / 2 * 100

        # Emotion breakdown
        emotion_counts = df["emotion"].value_counts(normalize=True).to_dict()
        fear_pct = emotion_counts.get("fear", 0)
        greed_pct = emotion_counts.get("greed", 0)
        # Adjust fear/greed index
        fear_greed = fear_greed * (1 - fear_pct) + fear_greed * (1 + greed_pct)
        fear_greed = float(np.clip(fear_greed, 0, 100))

        # Sentiment momentum: last 3h vs previous 21h
        now = df["timestamp"].max()
        recent = df[df["timestamp"] >= now - timedelta(hours=3)]["raw_score"].mean()
        older = df[df["timestamp"] < now - timedelta(hours=3)]["raw_score"].mean()
        momentum = float(recent - older) if not np.isnan(recent - older) else 0.0

        # Per-source breakdown
        sources = df.groupby("source")["raw_score"].mean().to_dict()

        return AggregatedSentiment(
            ticker=str(ticker),
            timestamp=now,
            bullish_score=float(bullish),
            fear_greed_index=fear_greed,
            sentiment_momentum=momentum,
            volume=len(df),
            engagement=int(df["engagement"].sum()),
            sources=sources,
        )
