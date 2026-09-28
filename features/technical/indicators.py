"""
features/technical/indicators.py
Generates 80+ technical indicators for a given OHLCV dataframe.
Uses ta-lib (C bindings) where available, pure-pandas fallback otherwise.
Designed to be side-effect free and deterministic.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib C library not found, using pure-Python fallback")

import ta


def compute_all_indicators(
    df: pd.DataFrame,
    include_candlestick: bool = True,
    include_fibonacci: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Given a DataFrame with columns [open, high, low, close, volume],
    returns the same DataFrame enriched with technical indicators.

    Critical: no lookahead bias — all indicators use only past data.
    """
    df = df.copy()
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    # ── Trend ──────────────────────────────────────────────────────────────

    # Simple and exponential moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = c.rolling(period).mean()
        df[f"ema_{period}"] = c.ewm(span=period, adjust=False).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ADX (Average Directional Index)
    df["adx"] = ta.trend.ADXIndicator(h, l, c, window=14).adx()
    df["adx_pos"] = ta.trend.ADXIndicator(h, l, c, window=14).adx_pos()
    df["adx_neg"] = ta.trend.ADXIndicator(h, l, c, window=14).adx_neg()

    # Aroon
    aroon = ta.trend.AroonIndicator(h, l, window=25)
    df["aroon_up"] = aroon.aroon_up()
    df["aroon_down"] = aroon.aroon_down()
    df["aroon_ind"] = aroon.aroon_indicator()

    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(h, l)
    df["ichimoku_a"] = ichimoku.ichimoku_a()
    df["ichimoku_b"] = ichimoku.ichimoku_b()
    df["ichimoku_base"] = ichimoku.ichimoku_base_line()
    df["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()

    # Parabolic SAR
    df["psar"] = ta.trend.PSARIndicator(h, l, c).psar()
    df["psar_up"] = ta.trend.PSARIndicator(h, l, c).psar_up()
    df["psar_down"] = ta.trend.PSARIndicator(h, l, c).psar_down()

    # ── Momentum ───────────────────────────────────────────────────────────

    # RSI at multiple windows
    for period in [7, 14, 21]:
        df[f"rsi_{period}"] = ta.momentum.RSIIndicator(c, window=period).rsi()

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    df["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()

    # CCI (Commodity Channel Index)
    df["cci"] = ta.trend.CCIIndicator(h, l, c).cci()

    # Rate of Change
    for period in [5, 10, 20]:
        df[f"roc_{period}"] = ta.momentum.ROCIndicator(c, window=period).roc()

    # TRIX
    df["trix"] = ta.trend.TRIXIndicator(c).trix()

    # Money Flow Index
    df["mfi"] = ta.volume.MFIIndicator(h, l, c, v).money_flow_index()

    # Ultimate Oscillator
    df["uo"] = ta.momentum.UltimateOscillator(h, l, c).ultimate_oscillator()

    # Awesome Oscillator
    df["ao"] = ta.momentum.AwesomeOscillatorIndicator(h, l).awesome_oscillator()

    # KAMA (Kaufman Adaptive Moving Average)
    df["kama"] = ta.momentum.KAMAIndicator(c).kama()

    # ── Volatility ─────────────────────────────────────────────────────────

    # Bollinger Bands (multiple window/std combos)
    for window, std in [(20, 2.0), (20, 1.0), (10, 1.5)]:
        tag = f"{window}_{int(std*10)}"
        bb = ta.volatility.BollingerBands(c, window=window, window_dev=std)
        df[f"bb_upper_{tag}"] = bb.bollinger_hband()
        df[f"bb_lower_{tag}"] = bb.bollinger_lband()
        df[f"bb_mid_{tag}"] = bb.bollinger_mavg()
        df[f"bb_pct_{tag}"] = bb.bollinger_pband()   # % position within bands
        df[f"bb_width_{tag}"] = bb.bollinger_wband()  # band width

    # ATR
    for period in [7, 14, 21]:
        df[f"atr_{period}"] = ta.volatility.AverageTrueRange(h, l, c, window=period).average_true_range()

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(h, l, c)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_pct"] = kc.keltner_channel_pband()

    # Donchian Channel
    dc = ta.volatility.DonchianChannel(h, l, c)
    df["dc_upper"] = dc.donchian_channel_hband()
    df["dc_lower"] = dc.donchian_channel_lband()
    df["dc_mid"] = dc.donchian_channel_mband()

    # Ulcer Index
    df["ulcer_index"] = ta.volatility.UlcerIndex(c).ulcer_index()

    # ── Volume ─────────────────────────────────────────────────────────────

    # On-Balance Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()

    # VWAP
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()

    # Chaikin MF
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()

    # Volume Price Trend
    df["vpt"] = ta.volume.VolumePriceTrendIndicator(c, v).volume_price_trend()

    # Ease of Movement
    df["eom"] = ta.volume.EaseOfMovementIndicator(h, l, v).ease_of_movement()
    df["sma_eom"] = ta.volume.EaseOfMovementIndicator(h, l, v).sma_ease_of_movement()

    # Negative Volume Index
    df["nvi"] = ta.volume.NegativeVolumeIndexIndicator(c, v).negative_volume_index()

    # Volume rolling stats
    for period in [5, 10, 20]:
        df[f"vol_sma_{period}"] = v.rolling(period).mean()
        df[f"vol_ratio_{period}"] = v / df[f"vol_sma_{period}"]

    # ── Price-derived features ─────────────────────────────────────────────

    # Returns
    df["return_1d"] = c.pct_change(1)
    df["return_5d"] = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)
    df["return_20d"] = c.pct_change(20)

    # Log returns (more stationary, better for ML)
    df["log_return_1d"] = np.log(c / c.shift(1))
    df["log_return_5d"] = np.log(c / c.shift(5))
    df["log_return_20d"] = np.log(c / c.shift(20))

    # Rolling volatility (realised)
    for period in [5, 10, 20, 60]:
        df[f"realised_vol_{period}"] = df["log_return_1d"].rolling(period).std() * np.sqrt(252)

    # Gap analysis
    df["gap"] = (o - c.shift(1)) / c.shift(1)
    df["intraday_range"] = (h - l) / c
    df["upper_shadow"] = (h - np.maximum(o, c)) / (h - l + 1e-8)
    df["lower_shadow"] = (np.minimum(o, c) - l) / (h - l + 1e-8)
    df["body_size"] = abs(c - o) / (h - l + 1e-8)

    # Distance from moving averages (normalised)
    for ma in [20, 50, 200]:
        df[f"dist_sma_{ma}"] = (c - df[f"sma_{ma}"]) / df[f"sma_{ma}"]

    # 52-week high/low
    df["high_52w"] = h.rolling(252).max()
    df["low_52w"] = l.rolling(252).min()
    df["dist_52w_high"] = (c - df["high_52w"]) / df["high_52w"]
    df["dist_52w_low"] = (c - df["low_52w"]) / df["low_52w"]

    # ── Time/calendar features ─────────────────────────────────────────────

    if hasattr(df.index, "dayofweek"):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_quarter_start"] = df.index.is_quarter_start.astype(int)

        # Cyclic encoding to avoid discontinuities
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Candlestick patterns (via TA-Lib) ─────────────────────────────────

    if include_candlestick and TALIB_AVAILABLE:
        o_np = o.values
        h_np = h.values
        l_np = l.values
        c_np = c.values

        patterns = {
            "doji": talib.CDLDOJI,
            "hammer": talib.CDLHAMMER,
            "hanging_man": talib.CDLHANGINGMAN,
            "shooting_star": talib.CDLSHOOTINGSTAR,
            "engulfing": talib.CDLENGULFING,
            "harami": talib.CDLHARAMI,
            "morning_star": talib.CDLMORNINGSTAR,
            "evening_star": talib.CDLEVENINGSTAR,
            "three_white_soldiers": talib.CDL3WHITESOLDIERS,
            "three_black_crows": talib.CDL3BLACKCROWS,
            "dark_cloud_cover": talib.CDLDARKCLOUDCOVER,
            "piercing": talib.CDLPIERCING,
            "spinning_top": talib.CDLSPINNINGTOP,
            "marubozu": talib.CDLMARUBOZU,
        }
        for name, func in patterns.items():
            df[f"cdl_{name}"] = func(o_np, h_np, l_np, c_np)

    # ── Fibonacci retracement levels ───────────────────────────────────────

    if include_fibonacci:
        window = 20
        recent_high = h.rolling(window).max()
        recent_low = l.rolling(window).min()
        rng = recent_high - recent_low
        df["fib_0"] = recent_low
        df["fib_236"] = recent_low + 0.236 * rng
        df["fib_382"] = recent_low + 0.382 * rng
        df["fib_500"] = recent_low + 0.500 * rng
        df["fib_618"] = recent_low + 0.618 * rng
        df["fib_786"] = recent_low + 0.786 * rng
        df["fib_100"] = recent_high

        # How far is price from each level?
        for lvl in [236, 382, 500, 618, 786]:
            col = f"fib_{lvl}"
            df[f"dist_fib_{lvl}"] = (c - df[col]) / c

    # ── Z-score normalisation of key indicators ────────────────────────────

    norm_cols = ["rsi_14", "macd", "cci", "ao", "mfi"]
    for col in norm_cols:
        if col in df.columns:
            mu = df[col].rolling(252).mean()
            sigma = df[col].rolling(252).std()
            df[f"{col}_zscore"] = (df[col] - mu) / (sigma + 1e-8)

    if drop_na:
        # Drop rows with all-NaN indicators (warm-up period)
        df = df.dropna(subset=["sma_200", "rsi_14", "macd"])

    return df


def compute_cross_sectional_features(
    price_data: dict[str, pd.DataFrame],
    reference_ticker: str = "SPY",
) -> dict[str, pd.DataFrame]:
    """
    Compute features that require looking across multiple tickers:
    - Beta vs SPY
    - Relative strength vs sector
    - Rolling correlation with market
    """
    spy = price_data.get(reference_ticker)
    if spy is None:
        return price_data

    spy_ret = spy["close"].pct_change()
    result = {}

    for ticker, df in price_data.items():
        df = df.copy()
        ret = df["close"].pct_change()

        # Align on common dates
        aligned = pd.concat([ret, spy_ret], axis=1, join="inner")
        aligned.columns = ["stock", "market"]

        # Rolling beta
        cov = aligned["stock"].rolling(60).cov(aligned["market"])
        var = aligned["market"].rolling(60).var()
        beta = cov / (var + 1e-10)

        # Rolling correlation
        corr = aligned["stock"].rolling(60).corr(aligned["market"])

        # Relative strength (stock performance / market performance)
        rs = (1 + aligned["stock"]).cumprod() / (1 + aligned["market"]).cumprod()

        df = df.join(pd.DataFrame({
            "beta": beta,
            "market_corr": corr,
            "relative_strength": rs,
        }), how="left")

        # Sharpe ratio (rolling 252-day)
        annual = ret.rolling(252).mean() * 252
        vol = ret.rolling(252).std() * np.sqrt(252)
        df["rolling_sharpe"] = annual / (vol + 1e-8)

        result[ticker] = df

    return result
