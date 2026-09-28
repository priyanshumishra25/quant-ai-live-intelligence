"""
Continuous retraining service.
Runs on a schedule (cron-style) and also responds to Kafka drift triggers.
Implements PSI-based feature drift detection and performance degradation checks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    ticker: str
    feature: str
    psi:     float
    drifted: bool
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    PSI > 0.25 → significant drift.
    PSI 0.1–0.25 → moderate drift.
    PSI < 0.1 → stable.
    """
    eps = 1e-8
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current,   bins=bins)

    ref_pct = (ref_counts + eps) / (len(reference) + eps)
    cur_pct = (cur_counts + eps) / (len(current)   + eps)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_feature_drift(
    baseline: pd.DataFrame,
    current:  pd.DataFrame,
    threshold: float = 0.25,
) -> list[DriftReport]:
    reports = []
    for col in baseline.columns:
        if col not in current.columns:
            continue
        psi = population_stability_index(
            baseline[col].dropna().values,
            current[col].dropna().values,
        )
        reports.append(DriftReport(
            ticker="global",
            feature=col,
            psi=psi,
            drifted=psi > threshold,
        ))
    return reports


# ---------------------------------------------------------------------------
# Performance monitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Tracks rolling directional accuracy and triggers retraining
    if accuracy falls below threshold.
    """

    def __init__(
        self,
        window:     int   = 20,
        min_acc:    float = 0.52,
        min_sharpe: float = 0.0,
    ):
        self._window    = window
        self._min_acc   = min_acc
        self._min_sharpe = min_sharpe
        self._history: dict[str, list[dict]] = {}

    def record(self, ticker: str, predicted_return: float, actual_return: float) -> None:
        buf = self._history.setdefault(ticker, [])
        buf.append({
            "pred":   predicted_return,
            "actual": actual_return,
            "hit":    int(np.sign(predicted_return) == np.sign(actual_return)),
        })
        if len(buf) > self._window * 3:
            buf.pop(0)

    def should_retrain(self, ticker: str) -> tuple[bool, str]:
        buf = self._history.get(ticker, [])
        if len(buf) < self._window:
            return False, "insufficient_data"

        recent = buf[-self._window:]
        acc    = np.mean([r["hit"] for r in recent])
        rets   = np.array([np.sign(r["pred"]) * r["actual"] for r in recent])
        sharpe = float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(252 / self._window))

        if acc < self._min_acc:
            return True, f"acc_drop:{acc:.3f}<{self._min_acc}"
        if sharpe < self._min_sharpe:
            return True, f"sharpe_drop:{sharpe:.2f}<{self._min_sharpe}"
        return False, "healthy"


# ---------------------------------------------------------------------------
# Retrainer
# ---------------------------------------------------------------------------

class RetrainingService:
    """
    Orchestrates all retraining jobs.
    - Scheduled daily at market close (4:30 PM ET)
    - On-demand triggered by Kafka `retraining.trigger` topic
    - Drift-triggered after PSI check
    """

    def __init__(
        self,
        data_provider:    Any,   # MarketDataOrchestrator
        feature_builder:  Any,   # FeatureBuilder from dynamic_ensemble.py
        xgb_trainer:      Any,   # XGBoostTrainer
        lgb_trainer:      Any,   # LightGBMTrainer
        seq_trainer:      Any,   # WalkForwardPipeline
        ensemble:         Any,   # DynamicEnsemble
        perf_monitor:     PerformanceMonitor | None = None,
        tickers:          list[str] | None = None,
        lookback_days:    int = 504,
    ):
        self.data_provider   = data_provider
        self.feature_builder = feature_builder
        self.xgb_trainer     = xgb_trainer
        self.lgb_trainer     = lgb_trainer
        self.seq_trainer     = seq_trainer
        self.ensemble        = ensemble
        self.monitor         = perf_monitor or PerformanceMonitor()
        self.tickers         = tickers or ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]
        self.lookback        = lookback_days
        self._baseline_features: dict[str, pd.DataFrame] = {}
        self._last_retrain: dict[str, datetime] = {}

    # -----------------------------------------------------------------------

    async def retrain_ticker(self, ticker: str, reason: str = "scheduled") -> bool:
        start = time.monotonic()
        logger.info("Retraining %s (reason=%s)", ticker, reason)

        try:
            # 1. Fetch fresh data
            end   = datetime.utcnow()
            start_date = end - timedelta(days=self.lookback)
            df    = await self.data_provider.fetch(ticker, start=start_date, end=end)
            if df is None or len(df) < 100:
                logger.warning("Insufficient data for %s", ticker)
                return False

            # 2. Build features
            X = self.feature_builder.build_features(df)
            y_ret = df["close"].pct_change().shift(-1).dropna()
            y_dir = (y_ret > 0).astype(int)
            X     = X.loc[y_ret.index]

            # 3. Check drift vs baseline
            if ticker in self._baseline_features:
                drift_reports = check_feature_drift(
                    self._baseline_features[ticker], X.tail(60)
                )
                drifted = [r for r in drift_reports if r.drifted]
                if drifted:
                    logger.warning(
                        "%s: %d features drifted (PSI > 0.25): %s",
                        ticker,
                        len(drifted),
                        [r.feature for r in drifted[:5]],
                    )

            # 4. Retrain XGBoost
            with mlflow.start_run(run_name=f"retrain_xgb_{ticker}"):
                self.xgb_trainer.fit(X, y_ret, y_dir)
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("reason", reason)
                mlflow.log_param("n_samples", len(X))

            # 5. Retrain LightGBM
            with mlflow.start_run(run_name=f"retrain_lgb_{ticker}"):
                self.lgb_trainer.fit(X, y_ret, y_dir)
                mlflow.log_param("ticker", ticker)

            # 6. Save baseline for drift detection
            self._baseline_features[ticker] = X.tail(252).copy()
            self._last_retrain[ticker] = datetime.utcnow()

            elapsed = time.monotonic() - start
            logger.info("Retrained %s in %.1fs", ticker, elapsed)
            return True

        except Exception as exc:
            logger.error("Retraining failed for %s: %s", ticker, exc, exc_info=True)
            return False

    async def retrain_all(self, reason: str = "scheduled") -> dict[str, bool]:
        results = {}
        for ticker in self.tickers:
            results[ticker] = await self.retrain_ticker(ticker, reason)
            await asyncio.sleep(2)   # don't hammer the data provider
        return results

    async def check_and_retrain(self) -> None:
        """Performance-driven retraining check."""
        for ticker in self.tickers:
            should, reason = self.monitor.should_retrain(ticker)
            if should:
                await self.retrain_ticker(ticker, reason=f"perf_drop:{reason}")

    # -----------------------------------------------------------------------
    # Kafka trigger listener
    # -----------------------------------------------------------------------

    async def listen_for_triggers(
        self,
        bootstrap_servers: str = "localhost:9092",
    ) -> None:
        from aiokafka import AIOKafkaConsumer
        import json

        consumer = AIOKafkaConsumer(
            "retraining.trigger",
            bootstrap_servers=bootstrap_servers,
            group_id="retraining-service",
            auto_offset_reset="latest",
        )
        await consumer.start()
        try:
            async for msg in consumer:
                payload = json.loads(msg.value)
                ticker  = payload.get("ticker")
                reason  = payload.get("reason", "kafka_trigger")
                if ticker:
                    await self.retrain_ticker(ticker, reason)
                else:
                    await self.retrain_all(reason)
        finally:
            await consumer.stop()

    # -----------------------------------------------------------------------
    # Scheduler setup
    # -----------------------------------------------------------------------

    def build_scheduler(self) -> AsyncIOScheduler:
        scheduler = AsyncIOScheduler()

        # Daily full retrain at 4:30 PM ET (market close)
        scheduler.add_job(
            self.retrain_all,
            CronTrigger(hour=16, minute=30, timezone="America/New_York"),
            kwargs={"reason": "daily_scheduled"},
            id="daily_retrain",
            replace_existing=True,
        )

        # Performance check every 4 hours during market hours
        scheduler.add_job(
            self.check_and_retrain,
            CronTrigger(
                hour="9-16", minute="0", timezone="America/New_York",
                day_of_week="mon-fri",
            ),
            id="perf_check",
            replace_existing=True,
        )

        return scheduler

    async def run(self, bootstrap_servers: str = "localhost:9092") -> None:
        scheduler = self.build_scheduler()
        scheduler.start()
        logger.info("Retraining service started")

        # Run Kafka listener and scheduler concurrently
        try:
            await asyncio.gather(
                self.listen_for_triggers(bootstrap_servers),
                asyncio.sleep(float("inf")),   # keep alive
            )
        finally:
            scheduler.shutdown()
