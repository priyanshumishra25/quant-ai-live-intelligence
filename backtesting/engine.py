"""
backtesting/engine.py
Walk-forward backtesting engine for research and strategy evaluation.
Includes realistic transaction costs, slippage, position sizing,
risk management, and comprehensive performance metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import get_settings

settings = get_settings()
trade_cfg = settings.trading


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    entry_date: date
    exit_date: Optional[date]
    side: str                  # "LONG" | "SHORT"
    entry_price: float
    exit_price: float = 0.0
    shares: float = 0.0
    entry_signal: str = ""
    exit_signal: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class PortfolioSnapshot:
    date: date
    equity: float
    cash: float
    positions_value: float
    daily_return: float
    drawdown: float
    num_positions: int


@dataclass
class BacktestResults:
    trades: list[Trade]
    snapshots: list[PortfolioSnapshot]
    equity_curve: pd.Series
    stats: dict


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizer
# ─────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """Implements Kelly Criterion and volatility-based position sizing."""

    def __init__(
        self,
        method: str = "kelly",
        max_pct: float = 0.05,
        kelly_fraction: float = 0.25,
    ):
        self.method = method
        self.max_pct = max_pct
        self.kelly_fraction = kelly_fraction

    def size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        volatility: float,
        signal_confidence: float = 1.0,
    ) -> float:
        """
        Returns position size as fraction of equity.
        Kelly: f* = (bp - q) / b  where b=odds, p=win_rate, q=1-p
        Fractional Kelly: multiply by kelly_fraction for risk control.
        """
        if self.method == "fixed":
            return min(self.max_pct * signal_confidence, self.max_pct)

        if avg_loss == 0:
            return self.max_pct * signal_confidence

        b = avg_win / avg_loss         # odds ratio
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0.0

        # Fractional Kelly
        kelly *= self.kelly_fraction

        if self.method == "vol_adjusted_kelly":
            # Scale down in high vol regimes
            vol_scalar = min(0.20 / (volatility + 1e-8), 1.0)
            kelly *= vol_scalar

        # Cap at max position size, scale by signal confidence
        return float(np.clip(kelly * signal_confidence, 0.0, self.max_pct))


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Enforces risk limits at portfolio and position level.
    Detects market regime changes and can halt trading.
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,
        max_var_95: float = 0.03,
        max_sector_pct: float = 0.20,
        max_positions: int = 20,
    ):
        self.max_drawdown = max_drawdown
        self.max_var_95 = max_var_95
        self.max_sector_pct = max_sector_pct
        self.max_positions = max_positions
        self._halted = False
        self._halt_reason = ""

    def check_can_trade(
        self,
        equity: float,
        peak_equity: float,
        current_var: float,
        n_positions: int,
    ) -> tuple[bool, str]:
        """Returns (can_trade, reason)."""
        if self._halted:
            return False, self._halt_reason

        drawdown = (peak_equity - equity) / peak_equity
        if drawdown >= self.max_drawdown:
            self._halted = True
            self._halt_reason = f"Max drawdown breached: {drawdown:.1%}"
            return False, self._halt_reason

        if current_var > self.max_var_95 and n_positions > 0:
            return False, f"VaR too high: {current_var:.1%}"

        if n_positions >= self.max_positions:
            return False, f"Max positions reached: {n_positions}"

        return True, ""

    def reset(self) -> None:
        """Reset halt state (e.g. for new backtest fold)."""
        self._halted = False
        self._halt_reason = ""

    def compute_portfolio_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """Historical VaR from recent return distribution."""
        if len(returns) < 10:
            return 0.0
        return float(np.abs(np.percentile(returns.dropna(), (1 - confidence) * 100)))


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Usage:
        engine = BacktestEngine(price_data, signal_generator)
        results = engine.run(
            train_size=504, step_size=63, n_splits=8
        )
    """

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],    # ticker → OHLCV df
        signal_generator: Callable,              # (df, ticker) → {signal, confidence, return_est}
        initial_equity: float = 1_000_000.0,
        commission_pct: float = trade_cfg.commission_pct,
        slippage_pct: float = trade_cfg.slippage_pct,
        short_borrow_rate: float = trade_cfg.short_borrow_rate,
    ):
        self.price_data = price_data
        self.signal_generator = signal_generator
        self.initial_equity = initial_equity
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.short_borrow_rate = short_borrow_rate
        self.sizer = PositionSizer(
            kelly_fraction=trade_cfg.kelly_fraction,
            max_pct=trade_cfg.max_position_pct,
        )
        self.risk_mgr = RiskManager(
            max_drawdown=trade_cfg.max_drawdown_pct,
        )

    def run(
        self,
        train_size: int = 504,
        step_size: int = 63,
        n_splits: int = 8,
        hold_days: int = 5,
    ) -> BacktestResults:
        """
        Walk-forward validation: train on [0..train_size], test on [train_size..train_size+step].
        Slide the window forward by step_size and repeat n_splits times.

        CRITICAL: The signal generator is re-fitted on each training window.
        Never use data from the test window during training.
        """
        all_trades: list[Trade] = []
        all_snapshots: list[PortfolioSnapshot] = []
        equity_series: dict[date, float] = {}

        # Portfolio state persists across contiguous out-of-sample folds.  Only
        # the model is re-fitted at each boundary; capital is not reset.
        equity = self.initial_equity
        cash = equity
        positions: dict[str, Trade] = {}
        peak_equity = equity
        self.risk_mgr.reset()
        recent_returns: list[float] = []
        previous_equity = equity
        win_count, loss_count = 0, 0
        avg_win, avg_loss = 0.02, 0.01

        # Build unified date index
        all_dates = sorted(set(
            d for df in self.price_data.values()
            for d in df.index.date
        ))

        for split_idx in range(n_splits):
            train_start = split_idx * step_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + step_size

            if test_end > len(all_dates):
                logger.info(f"Split {split_idx}: not enough data, stopping")
                break

            train_dates = all_dates[train_start:train_end]
            test_dates = all_dates[test_start:test_end]

            logger.info(
                f"Walk-forward split {split_idx+1}/{n_splits}: "
                f"train={train_dates[0]}..{train_dates[-1]}, "
                f"test={test_dates[0]}..{test_dates[-1]}"
            )

            # Re-train models on training window (the signal_generator handles this)
            train_data = self._slice(self.price_data, train_dates)
            try:
                self.signal_generator.fit(train_data)
            except Exception as e:
                logger.error(f"Signal generator fit failed on split {split_idx}: {e}")
                continue

            # Run simulation on the next contiguous out-of-sample window.
            for dt in test_dates:
                # Collect expiring positions (held for hold_days)
                exited_today = []
                for ticker, trade in list(positions.items()):
                    days_held = (dt - trade.entry_date).days
                    if days_held >= hold_days:
                        exit_price = self._get_price(ticker, dt)
                        if exit_price == 0.0:
                            continue
                        # Realistic exit with slippage
                        if trade.side == "LONG":
                            exit_price *= (1 - self.slippage_pct)
                        else:
                            exit_price *= (1 + self.slippage_pct)
                        exit_commission = exit_price * trade.shares * self.commission_pct
                        entry_notional = trade.entry_price * trade.shares
                        borrow = 0.0
                        if trade.side == "LONG":
                            gross_pnl = (exit_price - trade.entry_price) * trade.shares
                            cash += exit_price * trade.shares - exit_commission
                        else:
                            gross_pnl = (trade.entry_price - exit_price) * trade.shares
                            borrow = entry_notional * self.short_borrow_rate * days_held / 365
                            # A short sale credits proceeds at entry; closing debits the cover cost.
                            cash -= exit_price * trade.shares + exit_commission + borrow

                        trade.exit_date = dt
                        trade.exit_price = exit_price
                        trade.commission += exit_commission
                        trade.pnl = gross_pnl - trade.commission - borrow
                        trade.pnl_pct = trade.pnl / entry_notional if entry_notional else 0.0
                        pnl = trade.pnl

                        # Update win/loss stats
                        if pnl > 0:
                            win_count += 1
                            avg_win = (avg_win * (win_count - 1) + trade.pnl_pct) / win_count
                        else:
                            loss_count += 1
                            avg_loss = (avg_loss * (loss_count - 1) + abs(trade.pnl_pct)) / loss_count

                        all_trades.append(trade)
                        exited_today.append(ticker)

                for ticker in exited_today:
                    del positions[ticker]

                # Compute portfolio state
                pos_value = sum(
                    self._get_price(t, dt) * p.shares * (1 if p.side == "LONG" else -1)
                    for t, p in positions.items()
                )
                equity = cash + pos_value
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity

                # Return is measured against the prior marked-to-market equity.
                daily_ret = (equity / previous_equity - 1.0) if previous_equity else 0.0

                # Risk check before entering new positions
                var_95 = self.risk_mgr.compute_portfolio_var(
                    pd.Series(recent_returns[-60:])
                )
                can_trade, halt_reason = self.risk_mgr.check_can_trade(
                    equity, peak_equity, var_95, len(positions)
                )
                if not can_trade:
                    logger.warning(f"{dt}: Trading halted — {halt_reason}")
                    recent_returns.append(daily_ret)
                    equity_series[dt] = equity
                    previous_equity = equity
                    all_snapshots.append(PortfolioSnapshot(
                        dt, equity, cash, pos_value, daily_ret, drawdown, len(positions)
                    ))
                    continue

                # Generate signals for all tickers
                test_slice = self._slice_date(self.price_data, dt)
                win_rate = win_count / max(win_count + loss_count, 1)

                for ticker, ticker_df in test_slice.items():
                    if ticker in positions:
                        continue   # already in position

                    try:
                        sig = self.signal_generator.predict_single(ticker_df, ticker)
                    except Exception as e:
                        continue

                    if sig["signal"] not in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL"):
                        continue

                    price = self._get_price(ticker, dt)
                    if price == 0.0:
                        continue

                    vol = float(ticker_df.get("realised_vol_20", pd.Series([0.2])).iloc[-1])
                    pos_frac = self.sizer.size(
                        equity, win_rate, avg_win, avg_loss,
                        vol, sig["signal_confidence"]
                    )
                    value = equity * pos_frac
                    shares = value / price

                    # Realistic entry with slippage
                    side = "LONG" if "BUY" in sig["signal"] else "SHORT"
                    entry_price = price * (1 + self.slippage_pct) if side == "LONG" else price * (1 - self.slippage_pct)
                    commission = entry_price * shares * self.commission_pct
                    entry_notional = entry_price * shares
                    required_capital = entry_notional + commission

                    if required_capital > equity * 0.95:
                        continue   # conservative margin / capital constraint

                    if side == "LONG":
                        if required_capital > cash:
                            continue
                        cash -= required_capital
                    else:
                        # Short-sale proceeds are credited to cash; the equity value of
                        # the liability is represented by the negative position value.
                        cash += entry_notional - commission

                    positions[ticker] = Trade(
                        ticker=ticker,
                        entry_date=dt,
                        exit_date=None,
                        side=side,
                        entry_price=entry_price,
                        shares=shares,
                        entry_signal=sig["signal"],
                        commission=commission,
                    )

                # Re-mark after new entries so entry commissions are reflected in
                # same-day equity and in the stored return series.
                pos_value = sum(
                    self._get_price(t, dt) * p.shares * (1 if p.side == "LONG" else -1)
                    for t, p in positions.items()
                )
                equity = cash + pos_value
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
                daily_ret = (equity / previous_equity - 1.0) if previous_equity else 0.0
                recent_returns.append(daily_ret)
                equity_series[dt] = equity
                previous_equity = equity

                all_snapshots.append(PortfolioSnapshot(
                    dt, equity, cash, pos_value, daily_ret, drawdown, len(positions)
                ))

        equity_curve = pd.Series(equity_series)
        stats = self._compute_stats(equity_curve, all_trades)
        return BacktestResults(all_trades, all_snapshots, equity_curve, stats)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _slice(
        self,
        data: dict[str, pd.DataFrame],
        dates: list[date],
    ) -> dict[str, pd.DataFrame]:
        date_set = set(dates)
        return {
            tk: df[np.isin(df.index.date, list(date_set))]
            for tk, df in data.items()
        }

    def _slice_date(
        self,
        data: dict[str, pd.DataFrame],
        dt: date,
    ) -> dict[str, pd.DataFrame]:
        """Return all feature rows up to and including date dt."""
        return {
            tk: df[df.index.date <= dt]
            for tk, df in data.items()
            if not df[df.index.date <= dt].empty
        }

    def _get_price(self, ticker: str, dt: date) -> float:
        df = self.price_data.get(ticker)
        if df is None:
            return 0.0
        rows = df[df.index.date == dt]
        if rows.empty:
            # Use last known price (Friday → weekend)
            rows = df[df.index.date <= dt]
            if rows.empty:
                return 0.0
        return float(rows["close"].iloc[-1])

    def _compute_stats(
        self,
        equity: pd.Series,
        trades: list[Trade],
    ) -> dict:
        if equity.empty:
            return {}

        daily_ret = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_years = len(equity) / 252

        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        # Sharpe and Sortino (annualised, 0% risk-free for simplicity)
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-10)) * np.sqrt(252)
        downside = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / (downside.std() + 1e-10)) * np.sqrt(252)

        # Max drawdown
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd = float(drawdown.min())
        calmar = cagr / abs(max_dd + 1e-10)

        # Trade stats
        closed = [t for t in trades if t.exit_date is not None]
        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl <= 0]
        win_rate = len(winners) / max(len(closed), 1)
        avg_win_pnl = np.mean([t.pnl for t in winners]) if winners else 0.0
        avg_loss_pnl = np.mean([t.pnl for t in losers]) if losers else 0.0
        profit_factor = (
            sum(t.pnl for t in winners) / (abs(sum(t.pnl for t in losers)) + 1e-8)
        )

        # VaR
        var_95 = float(np.abs(np.percentile(daily_ret, 5))) if len(daily_ret) > 20 else 0.0
        var_99 = float(np.abs(np.percentile(daily_ret, 1))) if len(daily_ret) > 20 else 0.0

        return {
            "total_return_pct": round(total_return * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "var_95_pct": round(var_95 * 100, 3),
            "var_99_pct": round(var_99 * 100, 3),
            "total_trades": len(closed),
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_win_usd": round(avg_win_pnl, 2),
            "avg_loss_usd": round(avg_loss_pnl, 2),
            "total_commission_usd": round(sum(t.commission for t in closed), 2),
        }

    def monte_carlo(
        self,
        n_simulations: int = 1000,
        n_years: int = 3,
        results: Optional[BacktestResults] = None,
    ) -> dict:
        """
        Bootstrap Monte Carlo simulation from observed daily returns.
        Returns distribution of possible future equity curves.
        """
        if results is None:
            raise ValueError("Provide BacktestResults from run()")

        daily_ret = results.equity_curve.pct_change().dropna().values
        if len(daily_ret) < 20:
            return {}

        n_days = n_years * 252
        sims = np.zeros((n_simulations, n_days))

        for i in range(n_simulations):
            drawn = np.random.choice(daily_ret, size=n_days, replace=True)
            sims[i] = np.cumprod(1 + drawn)

        final = sims[:, -1]
        return {
            "median_return_pct": round((np.median(final) - 1) * 100, 2),
            "p5_return_pct": round((np.percentile(final, 5) - 1) * 100, 2),
            "p95_return_pct": round((np.percentile(final, 95) - 1) * 100, 2),
            "prob_positive_pct": round(float(np.mean(final > 1)) * 100, 2),
            "prob_double_pct": round(float(np.mean(final > 2)) * 100, 2),
            "expected_max_drawdown_pct": round(
                float(np.mean([
                    (1 - np.min(sims[i] / np.maximum.accumulate(sims[i])))
                    for i in range(n_simulations)
                ])) * 100, 2
            ),
        }
