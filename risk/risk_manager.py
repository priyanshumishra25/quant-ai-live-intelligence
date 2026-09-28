"""
Comprehensive risk management module.
Implements VaR/CVaR, Kelly Criterion, portfolio exposure management,
market regime detection, and Black Swan early warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market regimes
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    BULL_LOW_VOL   = "BULL_LOW_VOL"
    BULL_HIGH_VOL  = "BULL_HIGH_VOL"
    BEAR_LOW_VOL   = "BEAR_LOW_VOL"
    BEAR_HIGH_VOL  = "BEAR_HIGH_VOL"   # crisis
    SIDEWAYS       = "SIDEWAYS"


@dataclass
class RegimeState:
    regime:      Regime
    trend:       float     # rolling 60-day return
    volatility:  float     # rolling 20-day ann. vol
    vix_level:   float
    confidence:  float     # 0–1
    timestamp:   str = ""


class RegimeDetector:
    """
    Rule-based + statistical regime detector.
    Uses rolling returns, volatility, VIX, and the yield curve.
    """

    def __init__(
        self,
        trend_window:  int = 60,
        vol_window:    int = 20,
        vol_threshold: float = 0.20,   # annualised
    ):
        self.trend_window  = trend_window
        self.vol_window    = vol_window
        self.vol_threshold = vol_threshold

    def detect(
        self,
        prices:      pd.Series,
        vix:         pd.Series | None = None,
        yield_spread: pd.Series | None = None,   # 10y−2y
    ) -> RegimeState:
        returns  = prices.pct_change().dropna()
        trend    = returns.tail(self.trend_window).mean() * 252
        raw_vol  = returns.tail(self.vol_window).std() * np.sqrt(252)
        vix_now  = float(vix.iloc[-1]) if vix is not None and len(vix) else 18.0

        high_vol = raw_vol > self.vol_threshold or vix_now > 25
        bull     = trend > 0.05

        if bull and not high_vol:
            regime = Regime.BULL_LOW_VOL
        elif bull and high_vol:
            regime = Regime.BULL_HIGH_VOL
        elif not bull and not high_vol:
            regime = Regime.BEAR_LOW_VOL
        elif not bull and high_vol:
            regime = Regime.BEAR_HIGH_VOL
        else:
            regime = Regime.SIDEWAYS

        # Confidence: how far from thresholds
        trend_conf = min(abs(trend) / 0.2, 1.0)
        vol_conf   = min(abs(raw_vol - self.vol_threshold) / 0.1, 1.0)
        confidence = (trend_conf + vol_conf) / 2

        return RegimeState(
            regime=regime,
            trend=float(trend),
            volatility=float(raw_vol),
            vix_level=vix_now,
            confidence=float(confidence),
        )

    def detect_inversion(self, yield_spread: pd.Series) -> bool:
        """Yield curve inversion check (recession predictor)."""
        return bool(yield_spread.iloc[-1] < 0)


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    var_95:   float
    var_99:   float
    cvar_95:  float
    cvar_99:  float
    max_loss_1d: float
    volatility:  float
    skewness:    float
    kurtosis:    float


def compute_var(
    returns: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "historical",   # historical | parametric | cornish-fisher
) -> float:
    """
    Value-at-Risk computation.
    Returns the loss (positive number) at the given confidence level.
    """
    if method == "historical":
        return float(-np.percentile(returns, (1 - confidence_level) * 100))

    elif method == "parametric":
        mu  = returns.mean()
        sig = returns.std()
        z   = stats.norm.ppf(1 - confidence_level)
        return float(-(mu + z * sig))

    elif method == "cornish-fisher":
        # Adjusted for skewness and kurtosis
        mu   = returns.mean()
        sig  = returns.std()
        sk   = stats.skew(returns)
        ku   = stats.kurtosis(returns)   # excess kurtosis
        z    = stats.norm.ppf(1 - confidence_level)
        z_cf = (
            z
            + (z ** 2 - 1) * sk / 6
            + (z ** 3 - 3 * z) * ku / 24
            - (2 * z ** 3 - 5 * z) * sk ** 2 / 36
        )
        return float(-(mu + z_cf * sig))

    raise ValueError(f"Unknown VaR method: {method}")


def compute_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall): mean loss beyond VaR."""
    threshold = np.percentile(returns, (1 - confidence_level) * 100)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def full_risk_metrics(returns: np.ndarray) -> RiskMetrics:
    return RiskMetrics(
        var_95       = compute_var(returns, 0.95, "cornish-fisher"),
        var_99       = compute_var(returns, 0.99, "cornish-fisher"),
        cvar_95      = compute_cvar(returns, 0.95),
        cvar_99      = compute_cvar(returns, 0.99),
        max_loss_1d  = float(-returns.min()),
        volatility   = float(returns.std() * np.sqrt(252)),
        skewness     = float(stats.skew(returns)),
        kurtosis     = float(stats.kurtosis(returns)),
    )


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_prob:   float,
    win_return: float,
    loss_return: float,
    fraction:   float = 0.5,   # fractional Kelly (recommended: 0.25–0.5)
) -> float:
    """
    Full Kelly: f = (p * b - q) / b
    where p = win_prob, b = win_return / |loss_return|, q = 1 - p.
    Fractional Kelly reduces variance at the cost of expected growth.
    """
    if loss_return >= 0 or win_return <= 0:
        return 0.0
    b = win_return / abs(loss_return)
    q = 1 - win_prob
    f = (win_prob * b - q) / b
    return float(max(0.0, min(f * fraction, 0.25)))   # cap at 25% per position


def kelly_from_returns(
    returns: np.ndarray,
    fraction: float = 0.5,
) -> float:
    """Estimate Kelly from historical return distribution."""
    wins  = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    win_prob   = len(wins) / len(returns)
    win_return = wins.mean()
    loss_return = losses.mean()
    return kelly_fraction(win_prob, win_return, loss_return, fraction)


# ---------------------------------------------------------------------------
# Portfolio optimisation (mean-variance + risk parity)
# ---------------------------------------------------------------------------

def mean_variance_weights(
    expected_returns: np.ndarray,
    cov_matrix:       np.ndarray,
    risk_aversion:    float = 2.0,
    long_only:        bool  = True,
    max_weight:       float = 0.30,
) -> np.ndarray:
    """
    Markowitz mean-variance optimisation.
    Maximises: μᵀw − (λ/2) wᵀΣw
    """
    n = len(expected_returns)

    def neg_utility(w: np.ndarray) -> float:
        ret  = expected_returns @ w
        var  = w @ cov_matrix @ w
        return -(ret - risk_aversion / 2 * var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0 if long_only else -0.20, max_weight)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_utility, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        logger.warning("Mean-variance optimisation failed: %s", result.message)
        return w0
    return result.x


def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Risk parity: each asset contributes equally to portfolio volatility.
    """
    n = cov_matrix.shape[0]

    def risk_contrib_diff(w: np.ndarray) -> float:
        w = np.abs(w)
        port_vol = np.sqrt(w @ cov_matrix @ w)
        marginal = cov_matrix @ w
        contrib  = w * marginal / (port_vol + 1e-10)
        target   = port_vol / n
        return float(np.sum((contrib - target) ** 2))

    result = minimize(
        risk_contrib_diff,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.01, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    w = np.abs(result.x)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Black swan / anomaly early warning
# ---------------------------------------------------------------------------

@dataclass
class BlackSwanWarning:
    level:       str     # LOW | MEDIUM | HIGH | CRITICAL
    indicators:  list[str]
    score:       float   # 0–1


def black_swan_score(
    current_returns: np.ndarray,
    vix:             float,
    credit_spread:   float | None = None,
    yield_spread:    float | None = None,
) -> BlackSwanWarning:
    """
    Multi-indicator Black Swan early-warning system.
    Aggregates tail-risk signals into a composite score.
    """
    indicators = []
    score = 0.0

    # Extreme return tail
    if len(current_returns) >= 20:
        recent_vol  = current_returns[-20:].std() * np.sqrt(252)
        hist_vol    = current_returns.std() * np.sqrt(252)
        vol_ratio   = recent_vol / (hist_vol + 1e-8)
        if vol_ratio > 2.0:
            indicators.append(f"vol_spike:{vol_ratio:.1f}x")
            score += min(vol_ratio / 5, 0.3)

        # 3-sigma daily moves
        z_score = abs(current_returns[-1] / (current_returns[-20:].std() + 1e-8))
        if z_score > 3:
            indicators.append(f"3sigma_move:z={z_score:.1f}")
            score += min(z_score / 10, 0.25)

    # VIX levels
    if vix > 40:
        indicators.append(f"VIX_critical:{vix:.0f}")
        score += 0.30
    elif vix > 25:
        indicators.append(f"VIX_elevated:{vix:.0f}")
        score += 0.15

    # Credit spreads widening
    if credit_spread is not None and credit_spread > 500:
        indicators.append(f"credit_spread:{credit_spread:.0f}bps")
        score += 0.20

    # Yield curve inversion
    if yield_spread is not None and yield_spread < -0.5:
        indicators.append(f"yield_inversion:{yield_spread:.2f}%")
        score += 0.15

    score = min(score, 1.0)
    level = (
        "CRITICAL" if score > 0.7
        else "HIGH"     if score > 0.45
        else "MEDIUM"   if score > 0.25
        else "LOW"
    )
    return BlackSwanWarning(level=level, indicators=indicators, score=score)


# ---------------------------------------------------------------------------
# Position sizer (integrates regime + Kelly + VaR)
# ---------------------------------------------------------------------------

@dataclass
class PositionDecision:
    ticker:      str
    signal:      str      # BUY | SELL | HOLD
    raw_size:    float    # $ amount before risk scaling
    risk_scaled: float    # final $ amount
    kelly:       float
    var_budget:  float    # allowed $ VaR
    regime_mult: float    # multiplier from current regime


REGIME_MULTIPLIERS: dict[Regime, float] = {
    Regime.BULL_LOW_VOL:  1.00,
    Regime.BULL_HIGH_VOL: 0.60,
    Regime.SIDEWAYS:      0.50,
    Regime.BEAR_LOW_VOL:  0.30,
    Regime.BEAR_HIGH_VOL: 0.15,
}


def size_position(
    ticker:          str,
    signal:          str,
    portfolio_value: float,
    returns:         np.ndarray,
    regime:          Regime,
    win_prob:        float = 0.55,
    max_position_pct: float = 0.10,
    var_limit_pct:   float = 0.02,   # 2% daily VaR budget
) -> PositionDecision:
    kelly = kelly_from_returns(returns)
    var95 = compute_var(returns, 0.95)

    # VaR-based sizing: solve for $ size such that VaR = var_limit
    var_budget = portfolio_value * var_limit_pct
    var_size   = var_budget / (var95 + 1e-8)

    # Kelly size
    kelly_size = portfolio_value * kelly

    # Take the minimum
    raw_size = min(var_size, kelly_size, portfolio_value * max_position_pct)

    # Apply regime multiplier
    reg_mult     = REGIME_MULTIPLIERS.get(regime, 0.5)
    risk_scaled  = raw_size * reg_mult

    return PositionDecision(
        ticker=ticker,
        signal=signal,
        raw_size=raw_size,
        risk_scaled=risk_scaled,
        kelly=kelly,
        var_budget=var_budget,
        regime_mult=reg_mult,
    )
