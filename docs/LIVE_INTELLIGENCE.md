# Live Intelligence Pipeline

## Inputs

`POST /api/v1/intelligence/analyze`

```json
{
  "query": "Apple",
  "horizon": "5d",
  "include_reddit_comments": true
}
```

`query` may be a company name or ticker supported by the configured market-data provider.

## 1. Resolution

`SYMBOL_SEARCH` resolves the input into a canonical symbol and company metadata. Exact ticker matches are preferred when present.

## 2. Market history

Daily OHLCV is normalised to:

```text
date index
open
high
low
close
volume
```

The model currently uses up to 1,500 daily observations.

## 3. News

The Alpha Vantage `NEWS_SENTIMENT` endpoint is queried for the resolved ticker. The system retains:

- title;
- summary;
- published time;
- source URL;
- ticker relevance;
- ticker sentiment score.

The full article body is not scraped.

## 4. Reddit

When OAuth credentials are configured, the service:

1. searches public Reddit posts for company name / ticker mentions;
2. paginates within the configured post budget;
3. ranks matched posts by engagement;
4. fetches top comments for a configurable number of high-engagement threads;
5. computes an ephemeral finance-oriented sentiment score;
6. returns only short snippets + source links.

It does not perform unsupported global comment search and does not crawl Reddit HTML.

## 5. Date alignment

Historical **news** items are mapped to the first market date on or after publication. Items later than the last historical market date are **not inserted into the historical training matrix**.

Reddit items are never inserted into the historical training matrix. For the current prediction, the latest seven-day news and Reddit evidence states are aggregated separately, allowing weekend/post-close information to influence the current forecast without creating historical look-ahead leakage or fitting model weights on Reddit content.

## 6. Aggregation

Historical news signals use relevance weighting and are smoothed with a short exponentially weighted window so old information decays rather than remaining indefinitely active. Current Reddit signals use engagement weighting but remain inference-only.

## 7. Model

The **fitted** model feature set is:

```text
ret_1d
ret_5d
ma_gap_5
ma_gap_20
vol_10
momentum_10
news_sentiment
news_volume
```

A regularised linear return model is fit for the requested future horizon.

The current Reddit state is intentionally **not a fitted feature set**. After the ridge forecast is produced, current Reddit sentiment is converted into a fixed, bounded overlay scaled by evidence strength. Reddit volume and engagement affect that evidence strength, but no coefficient is learned from Reddit content.

## 8. Chronological validation

Before final refitting, the latest portion of the eligible historical matrix is held out. The service reports:

- mean absolute error;
- directional accuracy;
- prediction/target correlation.

This is a diagnostic, not a claim of profitability.

## 9. Forecast output

The response includes:

- expected return and price;
- 95% residual interval;
- BUY/HOLD/SELL research-score probabilities;
- evidence-quality score;
- technical/news/Reddit contribution mix;
- top feature drivers;
- validation metrics;
- provider status/errors;
- attributed evidence items;
- recent price history.

## 10. Important limitations

- News coverage is limited by the provider plan and configured result budget.
- Reddit search coverage is limited by Reddit API capabilities and rate limits.
- Social sentiment can be manipulated or unrepresentative.
- Publication timing and exchange timezone semantics can matter around market close.
- The current lightweight social scorer is not a substitute for a validated financial-language model.
- Directional accuracy on one rolling holdout is not sufficient evidence of a tradable edge.
- The project is not an execution system.
