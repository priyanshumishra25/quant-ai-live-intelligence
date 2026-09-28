"""Market-data provider used by the runnable API.

The default provider is a bundled deterministic fixture so the repository works
without network access. Optional yfinance mode is supported when installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class MarketDataError(RuntimeError):
    pass


class MarketDataService:
    def __init__(self, provider: str, sample_dir: Path):
        self.provider = provider
        self.sample_dir = Path(sample_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def available_tickers(self) -> list[str]:
        if self.provider == "bundled":
            return sorted(p.stem.upper() for p in self.sample_dir.glob("*.csv"))
        return []

    def history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        ticker = ticker.upper()
        if self.provider == "bundled":
            return self._bundled(ticker).copy()
        if self.provider == "yfinance":
            try:
                import yfinance as yf  # type: ignore
            except ImportError as exc:
                raise MarketDataError("yfinance provider selected but yfinance is not installed") from exc
            df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
            if df.empty:
                raise MarketDataError(f"no market data returned for {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            needed = ["open", "high", "low", "close", "volume"]
            df = df[needed].dropna()
            df.index = pd.to_datetime(df.index)
            return df
        raise MarketDataError(f"unknown provider: {self.provider}")

    def latest_price(self, ticker: str) -> float:
        df = self.history(ticker)
        return float(df["close"].iloc[-1])

    def histories(self, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
        return {ticker.upper(): self.history(ticker) for ticker in tickers}

    def _bundled(self, ticker: str) -> pd.DataFrame:
        if ticker in self._cache:
            return self._cache[ticker]
        path = self.sample_dir / f"{ticker}.csv"
        if not path.exists():
            raise MarketDataError(
                f"{ticker} is not in the bundled fixture; available={','.join(self.available_tickers())}"
            )
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise MarketDataError(f"{path} missing columns: {sorted(missing)}")
        self._cache[ticker] = df
        return df
