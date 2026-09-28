"""Train the lightweight persisted inference artefacts for a chosen data provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_settings
from services.market_data import MarketDataService
from services.prediction_service import HORIZON_DAYS, fit_linear_artifact


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["bundled", "yfinance"], default=settings.data.market_data_provider)
    parser.add_argument("--tickers", nargs="+", default=settings.trading.equities)
    parser.add_argument("--output", type=Path, default=settings.model.artifact_dir)
    args = parser.parse_args()

    market = MarketDataService(args.provider, settings.data.sample_data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "provider": args.provider, "tickers": {}}
    for ticker in args.tickers:
        df = market.history(ticker)
        manifest["tickers"][ticker.upper()] = {}
        for horizon in HORIZON_DAYS:
            artifact = fit_linear_artifact(df, horizon)
            artifact.update({"ticker": ticker.upper(), "data_source": args.provider})
            path = args.output / f"{ticker.upper()}_{horizon}.json"
            path.write_text(json.dumps(artifact, indent=2) + "\n")
            manifest["tickers"][ticker.upper()][horizon] = path.name
            print(f"wrote {path}")
    (args.output / "registry.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
