"""Persisted lightweight inference and walk-forward signal utilities."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.market_data import MarketDataService

FEATURES = ["ret_1d", "ret_5d", "ma_gap_5", "ma_gap_20", "vol_10", "momentum_10"]
HORIZON_DAYS = {"1d": 1, "5d": 5, "20d": 20}


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ma_gap_5"] = close / close.rolling(5).mean() - 1.0
    out["ma_gap_20"] = close / close.rolling(20).mean() - 1.0
    out["vol_10"] = close.pct_change().rolling(10).std() * math.sqrt(252)
    out["momentum_10"] = close / close.shift(10) - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def fit_linear_artifact(df: pd.DataFrame, horizon: str, ridge: float = 1e-3) -> dict[str, Any]:
    days = HORIZON_DAYS[horizon]
    X = feature_frame(df)
    target = df["close"].shift(-days) / df["close"] - 1.0
    joined = X.assign(target=target).dropna()
    if len(joined) < 60:
        raise ValueError("not enough rows to fit bootstrap model")

    x = joined[FEATURES].to_numpy(dtype=float)
    y = joined["target"].to_numpy(dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds < 1e-12] = 1.0
    z = (x - means) / stds
    design = np.column_stack([np.ones(len(z)), z])
    reg = np.eye(design.shape[1]) * ridge
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ beta
    residual_std = float(np.std(y - pred, ddof=min(1, len(y) - 1)))
    return {
        "format_version": 1,
        "model_type": "ridge_linear_return",
        "horizon": horizon,
        "features": FEATURES,
        "feature_mean": means.tolist(),
        "feature_std": stds.tolist(),
        "intercept": float(beta[0]),
        "coefficients": beta[1:].tolist(),
        "residual_std": residual_std,
        "n_samples": int(len(joined)),
        "train_start": joined.index.min().date().isoformat(),
        "train_end": joined.index.max().date().isoformat(),
    }


@dataclass
class ArtifactRegistry:
    artifact_dir: Path

    def path_for(self, ticker: str, horizon: str) -> Path:
        return self.artifact_dir / f"{ticker.upper()}_{horizon}.json"

    def load(self, ticker: str, horizon: str) -> dict[str, Any]:
        path = self.path_for(ticker, horizon)
        if not path.exists():
            raise FileNotFoundError(f"no model artifact for {ticker.upper()} / {horizon}")
        return json.loads(path.read_text())

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.artifact_dir.glob("*.json") if p.name != "registry.json")


class PredictionService:
    def __init__(self, market: MarketDataService, registry: ArtifactRegistry):
        self.market = market
        self.registry = registry

    @property
    def loaded(self) -> bool:
        return bool(self.registry.available())

    def predict(self, ticker: str, horizon: str) -> dict[str, Any]:
        ticker = ticker.upper()
        if horizon not in HORIZON_DAYS:
            raise ValueError(f"unsupported horizon: {horizon}")
        df = self.market.history(ticker)
        artifact = self.registry.load(ticker, horizon)
        features = feature_frame(df).dropna()
        if features.empty:
            raise ValueError(f"not enough history for {ticker}")
        row = features.iloc[-1][artifact["features"]]
        x = row.to_numpy(dtype=float)
        mean = np.asarray(artifact["feature_mean"], dtype=float)
        std = np.asarray(artifact["feature_std"], dtype=float)
        coef = np.asarray(artifact["coefficients"], dtype=float)
        z = (x - mean) / std
        predicted_return = float(artifact["intercept"] + z @ coef)
        residual_std = max(float(artifact["residual_std"]), 1e-6)
        current_price = float(df["close"].iloc[-1])
        predicted_price = current_price * (1.0 + predicted_return)
        lower = current_price * (1.0 + predicted_return - 1.96 * residual_std)
        upper = current_price * (1.0 + predicted_return + 1.96 * residual_std)

        score = predicted_return / residual_std
        buy_raw = math.exp(max(min(score, 20), -20))
        sell_raw = math.exp(max(min(-score, 20), -20))
        hold_raw = math.exp(-abs(score) * 0.5)
        total = buy_raw + sell_raw + hold_raw
        buy_prob, sell_prob, hold_prob = buy_raw / total, sell_raw / total, hold_raw / total
        max_prob = max(buy_prob, sell_prob, hold_prob)
        if buy_prob == max_prob:
            signal = "STRONG_BUY" if buy_prob >= 0.70 else "BUY"
        elif sell_prob == max_prob:
            signal = "STRONG_SELL" if sell_prob >= 0.70 else "SELL"
        else:
            signal = "HOLD"

        contributions = z * coef
        total_abs = float(np.abs(contributions).sum()) or 1.0
        importance = {
            name: round(float(abs(value) / total_abs), 4)
            for name, value in zip(artifact["features"], contributions)
        }
        drivers = sorted(
            ((name, float(value)) for name, value in zip(artifact["features"], contributions)),
            key=lambda item: abs(item[1]), reverse=True,
        )[:3]
        explanation = "; ".join(
            f"{name} {'supports' if value >= 0 else 'opposes'} the forecast"
            for name, value in drivers
        )
        recent_returns = df["close"].pct_change().dropna().tail(60)
        vol = float(recent_returns.std() * math.sqrt(252)) if len(recent_returns) else 0.0
        var_95 = float(np.quantile(recent_returns, 0.05)) if len(recent_returns) >= 20 else 0.0
        risk_score = min(10.0, max(0.0, vol / 0.50 * 10.0))

        data_mode = "bundled_sample_model" if self.market.provider == "bundled" else "live_market_model"
        return {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_timestamp": df.index[-1].date().isoformat(),
            "horizon": horizon,
            "current_price": round(current_price, 4),
            "predicted_price": round(predicted_price, 4),
            "predicted_return_pct": round(predicted_return * 100.0, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "signal": signal,
            "signal_confidence": round(max_prob, 4),
            "buy_probability": round(buy_prob, 4),
            "sell_probability": round(sell_prob, 4),
            "hold_probability": round(hold_prob, 4),
            "predicted_volatility_pct": round(vol * 100.0, 3),
            "risk_score": round(risk_score, 3),
            "var_95": round(var_95, 6),
            "sentiment_score": 0.0,
            "explanation": explanation or "No material feature contribution.",
            "model_weights": {artifact["model_type"]: 1.0},
            "feature_importance": importance,
            "model_metadata": {
                "model_type": artifact["model_type"],
                "train_start": artifact["train_start"],
                "train_end": artifact["train_end"],
                "n_samples": artifact["n_samples"],
            },
            "data_mode": data_mode,
        }


class WalkForwardLinearSignal:
    """A small signal generator compatible with BacktestEngine."""
    def __init__(self, horizon: str = "5d"):
        self.horizon = horizon
        self.artifacts: dict[str, dict[str, Any]] = {}

    def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
        self.artifacts = {}
        for ticker, df in train_data.items():
            if len(df) >= 80:
                try:
                    self.artifacts[ticker] = fit_linear_artifact(df, self.horizon)
                except ValueError:
                    pass

    def predict_single(self, ticker_df: pd.DataFrame, ticker: str) -> dict[str, Any]:
        artifact = self.artifacts.get(ticker)
        if not artifact:
            return {"signal": "HOLD", "signal_confidence": 0.0, "return_est": 0.0}
        f = feature_frame(ticker_df).dropna()
        if f.empty:
            return {"signal": "HOLD", "signal_confidence": 0.0, "return_est": 0.0}
        x = f.iloc[-1][artifact["features"]].to_numpy(dtype=float)
        z = (x - np.asarray(artifact["feature_mean"])) / np.asarray(artifact["feature_std"])
        pred = float(artifact["intercept"] + z @ np.asarray(artifact["coefficients"]))
        sigma = max(float(artifact["residual_std"]), 1e-6)
        strength = abs(pred) / sigma
        confidence = float(min(0.95, 0.50 + 0.15 * strength))
        if pred > sigma * 0.15:
            signal = "STRONG_BUY" if strength > 1.0 else "BUY"
        elif pred < -sigma * 0.15:
            signal = "STRONG_SELL" if strength > 1.0 else "SELL"
        else:
            signal = "HOLD"
        return {"signal": signal, "signal_confidence": confidence, "return_est": pred}
