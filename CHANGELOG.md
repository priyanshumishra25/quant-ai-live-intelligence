# Changelog

## 3.0.2 — Temporal validation and repository cleanup

- add a horizon-length purge gap before the chronological holdout;
- estimate live forecast intervals from holdout residuals rather than final-fit in-sample residuals;
- select ridge alpha from the training block only via generalized cross-validation;
- replace BUY/HOLD/SELL pseudo-probabilities with explicitly heuristic bullish/neutral/bearish scores;
- report holdout size, approximate non-overlapping observations and validation reliability;
- align historical news to US market close semantics and drop malformed timestamps;
- document the fixed Reddit overlay as a design guardrail rather than an optimized coefficient;
- reject demo/default authentication secrets in staging/production;
- remove disconnected research prototypes (RL, GNN, Kafka, scraper, options/tree stacks, retrainer, legacy DB/research scaffolding);
- remove interview-specific repository copy and refresh publication metadata.

## 3.0.1 — Free-tier provider hardening

- Removed automatic live analysis on page load; Alpha Vantage is contacted only after an explicit Analyze action.
- Changed Alpha Vantage daily-history requests to `outputsize=compact` by default; `full` remains opt-in for premium keys.
- Added per-provider pacing so Alpha Vantage calls are serialized and spaced by 1.10 seconds by default.
- Added provider-level caches for symbol resolution (7 days), daily history (6 hours), news (30 minutes), and Reddit retrieval (10 minutes).
- Increased the final live-intelligence response cache to 30 minutes.
- Uppercase ticker inputs such as `AAPL` can skip the symbol-search API call.
- Changing forecast horizon reuses cached resolution/history/news instead of spending another Alpha Vantage request.
- Added friendly quota/rate-limit/premium errors and maps provider limit conditions to HTTP 429.
- Fixed historical news alignment so evidence older than the available market-history window is not collapsed onto the first training date.
- Added regression tests for provider caching, ticker-search skipping, free-tier compact history, and quota-message translation.

## 3.0.0 — Live Intelligence

### Added

- Arbitrary company/ticker resolution through Alpha Vantage `SYMBOL_SEARCH`.
- Live daily OHLCV retrieval through documented Alpha Vantage APIs.
- Market news + ticker relevance/sentiment ingestion.
- Reddit OAuth adapter with paginated public-post search and top-thread comment retrieval.
- Ephemeral finance-oriented Reddit sentiment scoring.
- Dated alternative-data feature construction with exponential decay.
- On-demand ridge model over technical + historical news features, with a fixed bounded Reddit inference-time overlay.
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
