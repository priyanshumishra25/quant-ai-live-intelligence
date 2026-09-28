"""Run a deterministic bundled-fixture experiment and export machine-readable results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from config import get_settings
from services.market_data import MarketDataService
from services.prediction_service import ArtifactRegistry, HORIZON_DAYS, PredictionService, WalkForwardLinearSignal


def run_backtest(df: pd.DataFrame, ticker: str, horizon: str, seed: int) -> dict:
    signal = WalkForwardLinearSignal(horizon)
    engine = BacktestEngine({ticker: df}, signal, initial_equity=100_000.0)
    results = engine.run(train_size=300, step_size=63, n_splits=6, hold_days=HORIZON_DAYS[horizon])
    np.random.seed(seed)
    mc = engine.monte_carlo(n_simulations=500, n_years=1, results=results) if len(results.equity_curve) >= 20 else {}
    return {
        "stats": results.stats,
        "monte_carlo": mc,
        "equity_curve": [
            {"date": dt.isoformat(), "equity": round(float(v), 2)}
            for dt, v in results.equity_curve.items()
        ],
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/sample_experiment"))
    parser.add_argument("--horizon", choices=list(HORIZON_DAYS), default="5d")
    args = parser.parse_args()

    market = MarketDataService("bundled", settings.data.sample_data_dir)
    predictor = PredictionService(market, ArtifactRegistry(settings.model.artifact_dir))
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "bundled_fixture_software_verification",
        "data_mode": "bundled_synthetic_fixture",
        "horizon": args.horizon,
        "warning": "Synthetic fixture results are software verification only, not empirical trading evidence.",
        "tickers": {},
    }
    stats_rows = []
    for ticker in market.available_tickers():
        pred = predictor.predict(ticker, args.horizon)
        bt = run_backtest(market.history(ticker), ticker, args.horizon, settings.model.seed)
        summary["tickers"][ticker] = {"prediction": pred, "backtest": bt}
        stats_rows.append({"ticker": ticker, **bt["stats"]})
        pd.DataFrame(bt["equity_curve"]).to_csv(args.output / f"{ticker}_{args.horizon}_equity.csv", index=False)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(stats_rows).to_csv(args.output / "backtest_stats.csv", index=False)
    print(f"wrote deterministic sample experiment to {args.output}")


if __name__ == "__main__":
    main()
