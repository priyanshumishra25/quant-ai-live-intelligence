"""Rebuild the deterministic bundled market fixture and lightweight model artefacts.

These data are synthetic and exist only so a clean checkout can exercise the
entire inference/backtest/API/frontend path without external network services.
They MUST NOT be reported as empirical market results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from services.prediction_service import HORIZON_DAYS, fit_linear_artifact

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample"
ARTIFACT_DIR = ROOT / "artifacts" / "models"
SEED = 20260920
TICKERS = {
    "AAPL": {"start": 132.0, "beta": 1.05, "idio": 0.010, "drift": 0.00035},
    "MSFT": {"start": 245.0, "beta": 0.95, "idio": 0.009, "drift": 0.00032},
    "NVDA": {"start": 42.0, "beta": 1.45, "idio": 0.016, "drift": 0.00045},
    "SPY":  {"start": 385.0, "beta": 0.70, "idio": 0.005, "drift": 0.00022},
    "QQQ":  {"start": 285.0, "beta": 0.90, "idio": 0.007, "drift": 0.00028},
}


def generate() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2022-07-01", periods=900)
    market_noise = rng.normal(0.0, 0.008, len(dates))
    regime = np.sin(np.linspace(0, 12 * np.pi, len(dates))) * 0.0012
    market = 0.00018 + regime + market_noise

    frames: dict[str, pd.DataFrame] = {}
    for idx, (ticker, cfg) in enumerate(TICKERS.items()):
        local = np.random.default_rng(SEED + idx + 1)
        returns = np.zeros(len(dates))
        eps = local.normal(0.0, cfg["idio"], len(dates))
        for i in range(1, len(dates)):
            momentum = 0.08 * returns[i - 1]
            mean_reversion = -0.03 * np.sum(returns[max(0, i - 5):i])
            returns[i] = cfg["drift"] + cfg["beta"] * market[i] + momentum + mean_reversion + eps[i]
            returns[i] = float(np.clip(returns[i], -0.12, 0.12))
        close = cfg["start"] * np.exp(np.cumsum(returns))
        overnight = local.normal(0, cfg["idio"] * 0.25, len(dates))
        open_ = close * np.exp(overnight)
        intraday = np.abs(local.normal(0.006, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + intraday)
        low = np.minimum(open_, close) * (1 - intraday)
        volume_base = 65_000_000 if ticker in {"AAPL", "SPY", "QQQ"} else 35_000_000
        volume = (volume_base * local.lognormal(0, 0.28, len(dates))).astype(int)
        df = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        frames[ticker] = df
    return frames


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frames = generate()
    registry: dict[str, object] = {
        "format_version": 1,
        "data_source": "deterministic synthetic research fixture",
        "seed": SEED,
        "tickers": {},
    }
    for ticker, raw in frames.items():
        path = DATA_DIR / f"{ticker}.csv"
        raw.to_csv(path, index=False, float_format="%.6f")
        df = raw.set_index("date")
        registry["tickers"][ticker] = {}
        for horizon in HORIZON_DAYS:
            artifact = fit_linear_artifact(df, horizon)
            artifact.update({
                "ticker": ticker,
                "data_source": "bundled_synthetic_fixture",
                "seed": SEED,
            })
            model_path = ARTIFACT_DIR / f"{ticker}_{horizon}.json"
            model_path.write_text(json.dumps(artifact, indent=2) + "\n")
            registry["tickers"][ticker][horizon] = model_path.name
    registry_path = ARTIFACT_DIR / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")

    published = sorted(DATA_DIR.glob("*.csv")) + sorted(ARTIFACT_DIR.glob("*.json"))
    manifest_lines = []
    for path in published:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "artifacts" / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")
    print(f"wrote {len(frames)} fixture files and {len(frames) * len(HORIZON_DAYS)} model artifacts")


if __name__ == "__main__":
    main()
