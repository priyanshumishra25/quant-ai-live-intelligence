#!/usr/bin/env python3
"""Run the live intelligence pipeline from the command line.

Examples:
  PYTHONPATH=. python scripts/analyze_stock.py Apple --horizon 5d
  PYTHONPATH=. python scripts/analyze_stock.py TSLA --no-comments --output results/tsla.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from config import get_settings
from services.live_intelligence import AlphaVantageClient, LiveIntelligenceService, RedditClient


async def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a stock using live market/news/Reddit evidence.")
    parser.add_argument("query", help="Company name or ticker, e.g. Apple or AAPL")
    parser.add_argument("--horizon", choices=("1d", "5d", "20d"), default="5d")
    parser.add_argument("--no-comments", action="store_true", help="Do not retrieve comments from matched Reddit threads")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    settings = get_settings()
    service = LiveIntelligenceService(
        AlphaVantageClient(settings.data.alpha_vantage_key),
        RedditClient(settings.data.reddit_client_id, settings.data.reddit_secret, settings.data.reddit_user_agent),
        lookback_days=settings.data.intelligence_lookback_days,
        news_fetch_limit=settings.data.intelligence_news_limit,
        reddit_fetch_limit=settings.data.intelligence_reddit_limit,
        reddit_comment_posts=settings.data.intelligence_reddit_comment_posts,
        reddit_comments_per_post=settings.data.intelligence_reddit_comments_per_post,
    )
    try:
        result = await service.analyze(args.query, args.horizon, include_reddit_comments=not args.no_comments)
    finally:
        await service.close()

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    asyncio.run(main())
