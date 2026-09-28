"""
models/ensemble/dynamic_ensemble.py
Dynamically weighted ensemble of all model types.
Weights are updated each day based on recent out-of-sample accuracy.
Includes full SHAP explainability pipeline.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
import torch
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from config import get_settings

settings = get_settings()
cfg = settings.model


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    ticker: str
    timestamp: datetime
    horizon: str               # "1d", "5d", "20d"

    # Price predictions
    predicted_price: float
    current_price: float
    predicted_return: float    # relative to current
    lower_bound: float         # confidence interval
    upper_bound: float

    # Signal
    signal: str                # "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
    signal_confidence: float   # 0.0 - 1.0
    buy_probability: float
    sell_probability: float
    hold_probability: float

    # Risk
    predicted_volatility: float
    risk_score: float          # 0 (low) - 10 (high)
    var_95: float              # 95% VaR

    # Metadata
    sentiment_score: float
    model_weights: dict[str, float]
    feature_importance: dict[str, float]
    explanation: str           # human-readable


# ─────────────────────────────────────────────────────────────────────────────
# Model Performance Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ModelPerformanceTracker:
    """
    Tracks each model's recent directional accuracy and RMSE.
    Used to compute dynamic ensemble weights.
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback = lookback_days
        self._records: defaultdict[str, list[dict]] = defaultdict(list)

    def record(
        self,
        model_name: str,
        predicted: float,
        actual: float,
        timestamp: datetime,
    ) -> None:
        self._records[model_name].append({
            "ts": timestamp,
            "predicted": predicted,
            "actual": actual,
        })
        # Prune old records
        cutoff = datetime.utcnow() - timedelta(days=self.lookback)
        self._records[model_name] = [
            r for r in self._records[model_name] if r["ts"] >= cutoff
        ]

    def get_weights(self, model_names: list[str]) -> dict[str, float]:
        """
        Compute softmax weights inversely proportional to recent RMSE.
        Models with lower recent error get higher weight.
        """
        scores = {}
        for name in model_names:
            records = self._records.get(name, [])
            if len(records) < 5:
                scores[name] = 1.0   # default until we have data
                continue
            preds = np.array([r["predicted"] for r in records])
            actuals = np.array([r["actual"] for r in records])
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            directional = np.mean(np.sign(preds) == np.sign(actuals))
            # Combined score: lower RMSE + higher directional = better
            scores[name] = directional / (rmse + 1e-8)

        # Normalise to sum to 1
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}

    def to_dict(self) -> dict:
        return {k: v for k, v in self._records.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Assembles the full feature matrix from:
      - Technical indicators
      - Sentiment scores
      - Macro data
      - Engineered features

    Returns a properly scaled numpy array ready for model input.
    """

    # Ordered list of all features used in training
    FEATURE_COLUMNS: list[str] = [
        # Price-derived
        "return_1d", "return_5d", "return_20d",
        "log_return_1d", "log_return_5d",
        "realised_vol_5", "realised_vol_20", "realised_vol_60",
        "gap", "intraday_range",
        # Trend
        "sma_20", "sma_50", "sma_200",
        "ema_20", "ema_50",
        "dist_sma_20", "dist_sma_50", "dist_sma_200",
        "macd", "macd_signal", "macd_hist",
        "adx", "adx_pos", "adx_neg",
        "aroon_ind",
        # Momentum
        "rsi_7", "rsi_14", "rsi_21",
        "stoch_k", "stoch_d",
        "williams_r",
        "cci",
        "roc_5", "roc_10", "roc_20",
        "mfi", "uo", "ao",
        # Volatility
        "atr_14", "atr_21",
        "bb_pct_20_20", "bb_width_20_20",
        "ulcer_index",
        # Volume
        "obv", "vwap", "cmf",
        "vol_ratio_5", "vol_ratio_20",
        # Cross-sectional
        "beta", "market_corr", "rolling_sharpe",
        # Sentiment
        "sentiment_score", "fear_greed_index", "sentiment_momentum",
        # Macro
        "fed_rate", "ten_yr_yield", "two_yr_yield",
        "vix", "dxy", "oil_wti",
        # Calendar
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_month_start", "is_month_end",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        X = self._select(df)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._select(df)
        if self._fitted:
            return self.scaler.transform(X)
        return X.values

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        missing = set(self.FEATURE_COLUMNS) - set(available)
        if missing:
            logger.warning(f"Missing features: {missing}")
        return df[available].fillna(0.0)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBuilder":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        fb = cls()
        fb.scaler = obj["scaler"]
        fb._fitted = obj["fitted"]
        return fb


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Predictor
# ─────────────────────────────────────────────────────────────────────────────

class DynamicEnsemble:
    """
    Combines predictions from:
      - LSTM (sequence model)
      - Transformer (TST)
      - XGBoost
      - LightGBM
      - CNN-LSTM
      - RL Agent (when in inference mode)

    Weights are dynamically adjusted based on recent performance.
    """

    MODEL_NAMES = ["lstm", "transformer", "xgboost", "lightgbm", "cnn_lstm"]

    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder
        self.tracker = ModelPerformanceTracker(cfg.ensemble_lookback_days)
        self.models: dict[str, Any] = {}
        self._current_weights: dict[str, float] = {
            m: 1 / len(self.MODEL_NAMES) for m in self.MODEL_NAMES
        }

    def register_model(self, name: str, model: Any) -> None:
        if name not in self.MODEL_NAMES:
            raise ValueError(f"Unknown model: {name}. Must be one of {self.MODEL_NAMES}")
        self.models[name] = model
        logger.info(f"Registered model: {name}")

    def predict(
        self,
        features: pd.DataFrame,
        ticker: str,
        current_price: float,
        sentiment: Optional[dict] = None,
        horizon: str = "1d",
    ) -> Prediction:
        """
        Generate a full prediction with confidence interval, signals, and explanations.
        """
        # Update weights from tracker
        self._current_weights = self.tracker.get_weights(list(self.models.keys()))

        # Build feature array
        X = self.feature_builder.transform(features)
        X_seq = self._build_sequence(X)    # (1, T, F) for sequence models
        X_flat = X[-1:, :]                  # (1, F) for tree models

        # Collect raw predictions
        raw_preds: dict[str, float] = {}
        raw_direction_probs: dict[str, np.ndarray] = {}

        for name, model in self.models.items():
            try:
                pred, dir_probs = self._predict_single(name, model, X_seq, X_flat)
                raw_preds[name] = pred
                raw_direction_probs[name] = dir_probs
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not raw_preds:
            raise RuntimeError("All models failed to produce predictions")

        # Weighted blend
        weights = {k: self._current_weights.get(k, 0.0) for k in raw_preds}
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}

        blended_return = sum(raw_preds[k] * weights[k] for k in raw_preds)

        # Blend direction probabilities
        if raw_direction_probs:
            blended_dir = sum(
                raw_direction_probs[k] * weights.get(k, 0.0)
                for k in raw_direction_probs
            )
        else:
            blended_dir = np.array([0.1, 0.6, 0.3])  # fallback: slight bias to hold

        buy_p, hold_p, sell_p = blended_dir[2], blended_dir[1], blended_dir[0]

        # Uncertainty: spread between model predictions as proxy
        pred_values = list(raw_preds.values())
        spread = np.std(pred_values) if len(pred_values) > 1 else abs(blended_return) * 0.1
        lower = blended_return - 1.96 * spread
        upper = blended_return + 1.96 * spread

        # Convert return to price
        predicted_price = current_price * (1 + blended_return)
        lower_price = current_price * (1 + lower)
        upper_price = current_price * (1 + upper)

        # Risk metrics
        vol = self._estimate_volatility(features)
        var_95 = current_price * blended_return - 1.645 * current_price * vol * np.sqrt(1/252)
        risk_score = self._compute_risk_score(vol, spread, blended_return)

        # Signal
        signal, confidence = self._generate_signal(
            blended_return, buy_p, sell_p, confidence_spread=spread
        )

        # Explainability (XGBoost SHAP values — fast and interpretable)
        feature_importance = {}
        if "xgboost" in self.models:
            try:
                feature_importance = self._compute_shap(X_flat)
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")

        sentiment_score = sentiment.get("bullish_score", 0.0) if sentiment else 0.0
        fear_greed = sentiment.get("fear_greed_index", 50.0) if sentiment else 50.0

        explanation = self._build_explanation(
            ticker, signal, blended_return, sentiment_score,
            fear_greed, feature_importance, weights
        )

        return Prediction(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            horizon=horizon,
            predicted_price=round(predicted_price, 2),
            current_price=round(current_price, 2),
            predicted_return=round(blended_return * 100, 3),
            lower_bound=round(lower_price, 2),
            upper_bound=round(upper_price, 2),
            signal=signal,
            signal_confidence=round(confidence, 3),
            buy_probability=round(float(buy_p), 3),
            sell_probability=round(float(sell_p), 3),
            hold_probability=round(float(hold_p), 3),
            predicted_volatility=round(vol * 100, 2),
            risk_score=round(risk_score, 1),
            var_95=round(var_95, 2),
            sentiment_score=round(sentiment_score, 3),
            model_weights={k: round(v, 3) for k, v in weights.items()},
            feature_importance={k: round(v, 4) for k, v in feature_importance.items()},
            explanation=explanation,
        )

    def _predict_single(
        self,
        name: str,
        model: Any,
        X_seq: np.ndarray,
        X_flat: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Dispatch to the correct model interface."""
        if name in ("lstm", "transformer", "cnn_lstm"):
            x_tensor = torch.FloatTensor(X_seq)
            with torch.no_grad():
                out = model(x_tensor)
            pred = float(out["predictions"][0, 0].cpu().numpy())
            if "direction_probs" in out:
                dir_p = out["direction_probs"][0].cpu().numpy()
            else:
                dir_p = np.array([0.1, 0.6, 0.3])
            return pred, dir_p

        elif name in ("xgboost", "lightgbm"):
            pred = float(model.predict(X_flat)[0])
            # Direction from predicted return sign
            if pred > 0.005:
                dir_p = np.array([0.1, 0.2, 0.7])
            elif pred < -0.005:
                dir_p = np.array([0.7, 0.2, 0.1])
            else:
                dir_p = np.array([0.15, 0.7, 0.15])
            return pred, dir_p

        raise ValueError(f"Unknown model type: {name}")

    def _build_sequence(self, X: np.ndarray) -> np.ndarray:
        """Shape feature matrix into (1, T, F) for sequence models."""
        T = min(cfg.lstm_seq_len, len(X))
        return X[-T:][np.newaxis, :, :]   # (1, T, F)

    def _estimate_volatility(self, features: pd.DataFrame) -> float:
        """Extract or compute realised volatility from features."""
        if "realised_vol_20" in features.columns:
            v = features["realised_vol_20"].iloc[-1]
            return float(v) if not np.isnan(v) else 0.20
        return 0.20   # fallback 20% annualised

    def _compute_risk_score(
        self,
        vol: float,
        model_spread: float,
        predicted_return: float,
    ) -> float:
        """0-10 risk score: high vol + high model disagreement = higher risk."""
        vol_score = min(vol / 0.40, 1.0) * 5      # normalise to 0-5
        uncertainty_score = min(model_spread / 0.02, 1.0) * 3   # 0-3
        adverse_score = max(-predicted_return / 0.05, 0.0) * 2  # 0-2
        return min(vol_score + uncertainty_score + adverse_score, 10.0)

    def _generate_signal(
        self,
        blended_return: float,
        buy_p: float,
        sell_p: float,
        confidence_spread: float,
    ) -> tuple[str, float]:
        """Map numeric predictions to trading signals with confidence."""
        confidence = 1.0 - min(confidence_spread / 0.05, 0.9)

        if blended_return > 0.03 and buy_p > 0.6:
            return "STRONG_BUY", confidence
        elif blended_return > 0.01 and buy_p > 0.5:
            return "BUY", confidence * 0.85
        elif blended_return < -0.03 and sell_p > 0.6:
            return "STRONG_SELL", confidence
        elif blended_return < -0.01 and sell_p > 0.5:
            return "SELL", confidence * 0.85
        else:
            return "HOLD", confidence * 0.7

    def _compute_shap(self, X_flat: np.ndarray) -> dict[str, float]:
        """Compute SHAP feature importance values for the XGBoost model."""
        model = self.models["xgboost"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_flat)
        feature_names = self.feature_builder.FEATURE_COLUMNS[:X_flat.shape[1]]
        importance = dict(zip(feature_names, np.abs(shap_values[0])))
        # Return top 10
        return dict(sorted(importance.items(), key=lambda x: -x[1])[:10])

    def _build_explanation(
        self,
        ticker: str,
        signal: str,
        predicted_return: float,
        sentiment: float,
        fear_greed: float,
        shap_features: dict[str, float],
        weights: dict[str, float],
    ) -> str:
        """Generate a human-readable explanation of the prediction."""
        lines = [
            f"Signal for {ticker}: {signal} ({predicted_return:+.2f}% expected return).",
        ]

        top_features = list(shap_features.keys())[:3]
        if top_features:
            lines.append(f"Key drivers: {', '.join(top_features)}.")

        if sentiment > 0.3:
            lines.append(f"Sentiment is strongly bullish (score: {sentiment:+.2f}).")
        elif sentiment < -0.3:
            lines.append(f"Sentiment is bearish (score: {sentiment:+.2f}).")

        fg_label = (
            "extreme fear" if fear_greed < 20 else
            "fear" if fear_greed < 40 else
            "neutral" if fear_greed < 60 else
            "greed" if fear_greed < 80 else
            "extreme greed"
        )
        lines.append(f"Market fear/greed: {fg_label} ({fear_greed:.0f}/100).")

        best_model = max(weights, key=weights.get)
        lines.append(f"Highest-weighted model: {best_model} ({weights[best_model]:.0%}).")

        return " ".join(lines)

    def update_tracker(
        self,
        model_name: str,
        predicted: float,
        actual: float,
        timestamp: datetime,
    ) -> None:
        self.tracker.record(model_name, predicted, actual, timestamp)
