"""Small, interpretable comparison models for the portfolio experiment lab.

These baselines deliberately stay lightweight so the default release can compare
an ML signal against non-ML heuristics without pulling in a deep-learning stack.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.prediction_service import WalkForwardLinearSignal


class MomentumSignal:
    """20-day trend baseline used as a transparent comparison model."""

    name = "momentum_baseline"

    def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
        return None

    def predict_single(self, ticker_df: pd.DataFrame, ticker: str) -> dict[str, Any]:
        if len(ticker_df) < 22:
            return {"signal": "HOLD", "signal_confidence": 0.0, "return_est": 0.0}
        close = ticker_df["close"]
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        daily_vol = float(close.pct_change().tail(20).std()) or 1e-6
        strength = abs(ret20) / max(daily_vol * np.sqrt(20), 1e-6)
        confidence = float(np.clip(0.50 + 0.12 * strength, 0.50, 0.90))
        if ret20 > daily_vol * 0.75:
            signal = "STRONG_BUY" if strength > 1.5 else "BUY"
        elif ret20 < -daily_vol * 0.75:
            signal = "STRONG_SELL" if strength > 1.5 else "SELL"
        else:
            signal = "HOLD"
        return {"signal": signal, "signal_confidence": confidence, "return_est": ret20 / 20.0}


class MeanReversionSignal:
    """20-day z-score baseline for contrast with trend-following behaviour."""

    name = "mean_reversion_baseline"

    def fit(self, train_data: dict[str, pd.DataFrame]) -> None:
        return None

    def predict_single(self, ticker_df: pd.DataFrame, ticker: str) -> dict[str, Any]:
        if len(ticker_df) < 22:
            return {"signal": "HOLD", "signal_confidence": 0.0, "return_est": 0.0}
        close = ticker_df["close"].tail(20)
        mean = float(close.mean())
        std = float(close.std()) or 1e-6
        z = float((close.iloc[-1] - mean) / std)
        confidence = float(np.clip(0.50 + 0.10 * abs(z), 0.50, 0.90))
        if z <= -1.0:
            signal = "STRONG_BUY" if z <= -2.0 else "BUY"
        elif z >= 1.0:
            signal = "STRONG_SELL" if z >= 2.0 else "SELL"
        else:
            signal = "HOLD"
        return {"signal": signal, "signal_confidence": confidence, "return_est": -z * 0.0025}


def build_signal(model: str, horizon: str):
    model = model.lower()
    if model == "ridge":
        return WalkForwardLinearSignal(horizon)
    if model == "momentum":
        return MomentumSignal()
    if model == "mean_reversion":
        return MeanReversionSignal()
    raise ValueError(f"unsupported experiment model: {model}")


MODEL_CATALOG = [
    {
        "id": "ridge",
        "label": "Ridge ML",
        "kind": "machine_learning",
        "description": "Walk-forward refitted ridge regression over engineered return/volatility features.",
    },
    {
        "id": "momentum",
        "label": "Momentum",
        "kind": "baseline",
        "description": "Transparent 20-day trend-following heuristic.",
    },
    {
        "id": "mean_reversion",
        "label": "Mean reversion",
        "kind": "baseline",
        "description": "Transparent 20-day price z-score heuristic.",
    },
]
