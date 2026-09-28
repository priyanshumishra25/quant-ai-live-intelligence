from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestEngine, PositionSizer


class FixedSignal:
    def __init__(self, signal: str):
        self.signal = signal

    def fit(self, _data):
        return self

    def predict_single(self, _df, _ticker):
        return {"signal": self.signal, "signal_confidence": 1.0}


def _prices(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return {"TEST": pd.DataFrame({"close": values, "realised_vol_20": 0.2}, index=idx)}


def _engine(values, signal):
    engine = BacktestEngine(
        _prices(values),
        FixedSignal(signal),
        initial_equity=100_000,
        commission_pct=0.0,
        slippage_pct=0.0,
        short_borrow_rate=0.0,
    )
    engine.sizer = PositionSizer(method="fixed", max_pct=0.10)
    return engine


def test_long_position_profits_in_rising_market():
    result = _engine([100, 100, 100, 110, 120], "BUY").run(
        train_size=2, step_size=3, n_splits=1, hold_days=1
    )
    assert result.trades
    assert result.equity_curve.iloc[-1] > 100_000


def test_short_position_profits_in_falling_market():
    result = _engine([120, 120, 120, 110, 100], "SELL").run(
        train_size=2, step_size=3, n_splits=1, hold_days=1
    )
    assert result.trades
    assert result.equity_curve.iloc[-1] > 100_000


def test_walk_forward_equity_does_not_reset_between_folds():
    values = [100, 100, 100, 105, 110, 115, 120, 125]
    result = _engine(values, "BUY").run(
        train_size=2, step_size=3, n_splits=2, hold_days=1
    )
    curve = result.equity_curve
    assert len(curve) == 6
    # A fold boundary must not jump back to the original $100k capital.
    assert curve.iloc[3] != 100_000
