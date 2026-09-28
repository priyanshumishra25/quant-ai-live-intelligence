# Architecture

## Goals

The default system has two intentionally separate execution paths:

1. a **live intelligence path** that demonstrates provider integration, alternative-data feature engineering and on-demand model serving; and
2. a **deterministic offline path** that keeps CI and clean checkouts reproducible without credentials or network access.

The paths share API/auth/cache/observability boundaries, but live-provider availability never determines whether the repository can be tested.

## Request flow — live analysis

```text
Browser
  │ POST /api/v1/intelligence/analyze
  ▼
FastAPI
  │ JWT + rate limit
  │ cache lookup
  ▼
LiveIntelligenceService
  │
  ├─ AlphaVantageClient
  │    ├─ SYMBOL_SEARCH
  │    ├─ TIME_SERIES_DAILY
  │    └─ NEWS_SENTIMENT
  │
  └─ RedditClient (optional)
       ├─ OAuth client token
       ├─ public post search
       └─ top comments for engaged matched threads
  │
  ▼
Evidence normalisation
  ├─ timestamps
  ├─ source/relevance
  ├─ sentiment
  └─ engagement
  │
  ▼
Feature construction
  ├─ technical price features
  ├─ news sentiment / volume
  └─ Reddit sentiment / volume / engagement (inference-only overlay)
  │
  ▼
Chronological holdout validation
  │
  ▼
Final on-demand ridge refit
  │
  ▼
Forecast
  ├─ return / target price
  ├─ interval
  ├─ signal distribution
  ├─ evidence quality
  ├─ source-family contributions
  └─ top feature drivers
  │
  ▼
Short attributed evidence stream + source links
```

## Provider isolation

External APIs are isolated behind adapters in `services/live_intelligence.py`. The modelling layer receives typed internal records and a pandas OHLCV frame; it does not know provider response schemas.

This gives three useful properties:

- provider payload changes are localised;
- providers can be mocked deterministically in tests;
- another licensed data provider can replace Alpha Vantage or Reddit without rewriting the model.

## Failure model

Historical market data is mandatory for a forecast. News and Reddit are **degradable sources**:

- if news fails, the fitted model can still use price-only features and current Reddit may contribute only through its bounded inference overlay;
- if Reddit is not configured, the fitted price+news model still runs and the response says Reddit is unavailable;
- if comment retrieval fails, already retrieved posts are retained;
- provider errors are returned in `sentiment.source_errors` rather than hidden.

The API never substitutes random/synthetic live evidence when a provider fails.

## Caching

`/api/v1/intelligence/analyze` is cached by normalized query, horizon and comment mode. The default final-response TTL is 30 minutes.

Provider data are cached below that layer as well: symbol resolution for 7 days, daily history for 6 hours, news for 30 minutes, and Reddit retrieval for 10 minutes. This means a horizon change can reuse the same upstream evidence without spending additional provider calls.

Local development can use the in-memory cache. Docker Compose uses Redis. Provider credentials remain server-side.

## Security boundaries

- Provider secrets are read from environment variables only.
- Browser clients never receive Alpha Vantage or Reddit credentials.
- Third-party titles/snippets are HTML-escaped before insertion into the dashboard.
- Third-party links are protocol-validated before use.
- JWT protects non-health API routes.
- Rate limiting protects the reference deployment from trivial provider-cost amplification.

## Scaling path

For a larger deployment I would change the architecture incrementally rather than pre-emptively adding infrastructure:

1. Redis-backed distributed rate limiting.
2. Provider-call caching keyed by resolved ticker rather than only final-response caching.
3. Queue live analyses when provider/model latency becomes material.
4. Persist only derived experiment metadata, not third-party user content, unless the provider licence explicitly permits retention.
5. Separate model-training workers from request-serving replicas.
6. Add a model registry with approval/rollback semantics.
7. Add trace IDs and OpenTelemetry across provider calls.
8. Add a licensed low-latency market-data provider if the product requirement becomes intraday/realtime trading rather than research.
