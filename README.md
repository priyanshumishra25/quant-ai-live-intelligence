# Quant AI — Live Stock Intelligence

**A software + AI engineering portfolio project that turns a company name or ticker into an auditable market forecast using historical price behaviour, current/historical market news, and Reddit discussion.**

The project is designed to demonstrate both sides of production AI work:

- **Software engineering:** typed client contracts, provider adapters, auth, caching, graceful degradation, observability, Docker, CI, source attribution and tests.
- **AI engineering:** feature pipelines, alternative-data aggregation, chronological validation, on-demand model fitting, uncertainty, explainability, provenance and reproducible offline evaluation.

> **Experimental research software, not investment advice.** A model signal is not a recommendation, and model confidence is not the probability of profit.

## What happens when you type `Apple`

```text
"Apple"
   │
   ├─► symbol resolution ─────────────► AAPL / Apple Inc.
   │
   ├─► historical OHLCV ──────────────► trend + technical features
   │
   ├─► market news ───────────────────► dated news sentiment + volume
   │                                         │
   │                                         ▼
   │                              chronological holdout validation
   │                                         │
   │                                         ▼
   │                               on-demand ridge model
   │                                         │
   └─► Reddit Data API ───────────────► current matched posts/comments
                                             │
                                             ▼
                                  fixed bounded social overlay
                                             │
                                             ▼
                     forecast + interval + probabilities + contributions
                                             │
                                             ▼
                              source-attributed evidence stream
```

The live response explains **which source family moved the forecast**, shows the top individual model drivers, reports chronological holdout metrics, and exposes the source items used to construct the current intelligence state.

## Product surfaces

### Live Intelligence

Search by company name or ticker. The dashboard displays:

- resolved company / symbol;
- historical trend and 1d/5d/20d/60d returns;
- 52-week range and volatility;
- market-news sentiment;
- Reddit sentiment and engagement;
- current evidence stream with source links;
- fused model forecast and 95% interval;
- BUY/HOLD/SELL research score distribution;
- technical vs news vs Reddit contribution mix;
- chronological validation metrics;
- provider degradation/errors instead of silently fabricating data.

### Offline Lab

The existing deterministic fixture path remains available with no API keys. It exercises persisted model loading, explainability, caching and cost-aware walk-forward evaluation in CI and clean checkouts.

### Experiment Lab

Compares the bundled Ridge ML signal with momentum and mean-reversion baselines under identical folds, holding periods, sizing rules, commission and slippage assumptions.

### System

Shows runtime telemetry, provider readiness, cache behaviour, architecture and implemented capabilities.

---

## Quick start

### 1. Reproducible offline mode

No external accounts are required:

```bash
cp .env.example .env
docker compose -f deployment/docker-compose.yml up --build
```

Open:

```text
Dashboard   http://localhost:3000
API docs    http://localhost:8000/api/docs
Health      http://localhost:8000/api/health
```

Local credentials:

```text
username: demo
password: quantai-demo
```

### 2. Enable live arbitrary-stock intelligence

Edit `.env`:

```dotenv
ALPHA_VANTAGE_KEY=your_key

# Optional Reddit evidence
REDDIT_CLIENT_ID=your_client_id
REDDIT_SECRET=your_client_secret
REDDIT_USER_AGENT=QuantAIResearch/3.0 by-your-reddit-username
```

The documented Alpha Vantage API is used for symbol search, daily OHLCV and market news/sentiment. Reddit evidence uses authenticated Reddit Data API access; no HTML scraping is used.

Restart:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

The **Live Intelligence** tab can then accept names such as `Apple`, `Tesla`, or supported symbols directly.

Provider plan and rate limits apply. The default retrieval budgets can be changed in `.env`:

```dotenv
INTELLIGENCE_LOOKBACK_DAYS=365
INTELLIGENCE_NEWS_LIMIT=500
INTELLIGENCE_REDDIT_LIMIT=100
INTELLIGENCE_REDDIT_COMMENT_POSTS=5
INTELLIGENCE_REDDIT_COMMENTS_PER_POST=20
REDIS_TTL_INTELLIGENCE=300
```

Reddit does not expose global comment search through the conventional Data API. The project therefore searches matched public posts and retrieves top comments from the most engaged matched threads. The UI states this coverage boundary explicitly.

### CLI

With live credentials configured:

```bash
make analyze QUERY="Apple" HORIZON=5d
```

or:

```bash
PYTHONPATH=. python scripts/analyze_stock.py TSLA --horizon 20d --output results/tsla.json
```

---

## Live prediction methodology

For a requested horizon (`1d`, `5d`, or `20d`), the live pipeline constructs:

### Technical features

```text
1-day return
5-day return
5-day moving-average gap
20-day moving-average gap
10-day annualised volatility
10-day momentum
```

### News features used for fitted model weights

```text
news sentiment
news volume
```

News uses Alpha Vantage's ticker-level relevance/sentiment where available. Historical news items are mapped to market dates and exponentially decayed so stale sentiment loses influence. Content after the last historical market date is **not backfilled into training history**; it is used only for the current forecast state.

### Reddit inference-only signals

```text
Reddit sentiment
Reddit volume
Reddit engagement
```

Reddit text is scored ephemerally with a transparent finance-oriented sentiment fallback in the lightweight serving path. **Reddit content never enters fitted model weights.** Instead, the latest Reddit state is converted into a small fixed, bounded inference-time overlay after the price+news ridge forecast is produced. The repository also retains a FinBERT research module outside the default dependency path.

The model then:

1. constructs a future-return target for the selected horizon;
2. standardises the features;
3. trains a regularised linear model on the chronological training segment;
4. evaluates on the most recent chronological holdout;
5. reports MAE, directional accuracy and prediction/target correlation;
6. refits on all eligible historical observations;
7. scores the current technical + news state;
8. applies a fixed, bounded Reddit inference overlay when Reddit evidence is available;
9. returns uncertainty bounds and feature/source contributions.

The model intentionally remains inspectable. The point of the project is not to hide weak evidence behind a huge neural network; it is to demonstrate a coherent end-to-end ML product and make every assumption discussable in an interview.

### Reddit data boundary

Reddit content is used **ephemerally at inference time only**. It is not inserted into the historical training matrix, does not enter fitted coefficients, and is not persisted as a training corpus or model artefact. The current Reddit aggregate affects the forecast only through a deliberately small fixed, bounded overlay. Live responses return short attributed snippets and source links rather than mirroring full content.

---

## API

### Live intelligence

```text
GET  /api/v1/intelligence/search?q=Apple
POST /api/v1/intelligence/analyze
```

Example:

```bash
curl -X POST http://localhost:8000/api/v1/intelligence/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apple",
    "horizon": "5d",
    "include_reddit_comments": true
  }'
```

The response contains:

```text
company
market trend metrics
news / Reddit sentiment summaries
source errors / degradation state
prediction + uncertainty + probabilities
evidence quality
source-family contribution mix
top model drivers
chronological validation metrics
source-attributed evidence items
recent price history
provider provenance
```

### Reproducible research API

```text
GET  /api/health
POST /api/v1/auth/token
GET  /api/v1/models
POST /api/v1/predict
GET  /api/v1/predict/{ticker}
POST /api/v1/predict/batch
GET  /api/v1/market/history/{ticker}
GET  /api/v1/market/overview
POST /api/v1/backtest
GET  /api/v1/backtest/{ticker}/equity.csv
GET  /api/v1/experiments/models
POST /api/v1/experiments/compare
GET  /api/v1/system
WS   /ws/predictions/{ticker}?token=...
WS   /ws/market/feed?token=...
GET  /metrics
```

---

## Architecture

```text
Strict TypeScript dashboard
          │
          ▼
       FastAPI
          │
     ┌────┴───────────────────────────────┐
     │                                    │
     ▼                                    ▼
Offline research path                Live intelligence path
persisted fixtures/models            provider adapter layer
     │                                    │
     ▼                               ┌────┴─────┐
walk-forward engine                  ▼          ▼
                               Alpha Vantage   Reddit OAuth API
                                      │          │
                                      └────┬─────┘
                                           ▼
                                dated feature aggregation
                                           ▼
                           price + news ridge + social overlay
                                           ▼
                          prediction + evidence + provenance
```

Redis can cache predictions and live intelligence responses. In-memory caching is the local zero-dependency fallback.

See:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/ENGINEERING.md`](docs/ENGINEERING.md)
- [`docs/LIVE_INTELLIGENCE.md`](docs/LIVE_INTELLIGENCE.md)
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)

---

## Verification

```bash
make verify
```

The release gate checks:

```text
Python compilation
16 unit/integration tests
Alpha Vantage adapter parsing via mocked documented responses
Reddit OAuth/search/comment adapter behaviour via mocked responses
live fusion pipeline with synthetic provider doubles
backtesting accounting regressions
API authentication and integration behaviour
strict TypeScript type-check
browser JavaScript syntax
fixture/model provenance and SHA checks
```

Live external-provider calls are deliberately **not** required by CI because CI must be deterministic and must not consume credentials/rate limits.

## Local development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
npm install --global typescript@5.8.3
cp .env.example .env
make verify
```

Run API:

```bash
make api
```

Run frontend in another terminal:

```bash
make frontend
```

Open `http://localhost:8080`.

Rebuild browser modules after changing TypeScript:

```bash
make frontend-build
```

---

## Repository map

```text
api/                       FastAPI REST/WebSocket surface
services/live_intelligence.py
                           provider adapters + evidence aggregation + live model
services/                  auth, cache, fixture inference and experiment engine
backtesting/               cost-aware walk-forward execution engine
artifacts/models/          persisted deterministic bootstrap models
data/sample/               deterministic OHLCV fixture
frontend/src/              strict TypeScript dashboard
frontend/dist/             committed browser-ready modules
models/                    optional heavier research models
features/                  technical + NLP research modules
risk/                      portfolio/risk utilities
deployment/                Docker, Compose, K8s and Prometheus config
scripts/analyze_stock.py   live CLI entry point
scripts/                   bootstrap, experiments and release checks
tests/                     deterministic unit + integration suite
docs/                      architecture, trade-offs and model documentation
```

## What this project claims — and does not claim

It **does** demonstrate that I can design and ship an ML-backed software product that integrates external data providers, turns unstructured evidence into features, validates models chronologically, exposes uncertainty, attributes sources, handles partial provider failure, and remains testable without network access.

It **does not** claim that this particular forecasting model produces durable real-market alpha. Establishing that would require a separately frozen empirical protocol with licensed point-in-time data, survivorship/corporate-action policy, a truly untouched holdout, hyperparameter-selection disclosure, ablations, repeated experiments, transaction-cost sensitivity and statistical controls.

## Security

Never commit `.env`, provider credentials, JWT secrets or private datasets. Change the demo auth credentials before exposing a deployment publicly. See [`SECURITY.md`](SECURITY.md).

## Licence

No reuse licence is selected automatically. Add the licence you intend to publish under before making the repository public.
