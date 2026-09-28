# Changelog

## 3.0.0 — Live Intelligence

### Added

- Arbitrary company/ticker resolution through Alpha Vantage `SYMBOL_SEARCH`.
- Live daily OHLCV retrieval through documented Alpha Vantage APIs.
- Market news + ticker relevance/sentiment ingestion.
- Reddit OAuth adapter with paginated public-post search and top-thread comment retrieval.
- Ephemeral finance-oriented Reddit sentiment scoring.
- Dated alternative-data feature construction with exponential decay.
- On-demand ridge fusion model combining technical, news and Reddit features.
- Chronological holdout validation before final refit.
- Forecast intervals, evidence-quality scoring and source-family contribution attribution.
- Source-attributed evidence stream in the dashboard.
- Live provider readiness and degradation reporting.
- `/api/v1/intelligence/search` and `/api/v1/intelligence/analyze` endpoints.
- `scripts/analyze_stock.py` and `make analyze QUERY=...` CLI workflow.
- Deterministic mocked-adapter tests for Alpha Vantage and Reddit.
- Live-intelligence architecture/methodology documentation.

### Changed

- Product framing now targets SWE + AI Engineering roles rather than a research-only platform.
- Dashboard defaults to Live Intelligence while retaining Offline Lab, Experiment Lab and System surfaces.
- Release version bumped to `3.0.0`.
- Docker Compose now passes provider credentials and retrieval-budget settings to the API container.
- Third-party source text is escaped before display and source URLs are protocol-validated.

### Preserved

- deterministic offline fixture path;
- persisted model registry;
- walk-forward backtesting/accounting fixes;
- experiment-vs-baseline lab;
- JWT auth, caching, WebSockets, metrics and containers.

## 2.1.0 — Engineering Portfolio

Introduced the strict-TypeScript product UI, Experiment Lab, System telemetry, engineering trade-off documentation and model card.

## 2.0.0 — Reproducible Research Platform

Converted the original prototype into a runnable, testable research-software release with corrected backtesting and deterministic fixtures.
