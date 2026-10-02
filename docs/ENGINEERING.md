# Engineering Decisions & Trade-offs

## Why this is not “just call an LLM with some headlines”

The product keeps the evidence and model path inspectable:

- provider data are normalized into explicit internal records;
- sentiment becomes dated numerical features;
- price and historical news features are joined chronologically;
- validation uses a later holdout, separated from training by a horizon-length purge gap, rather than a random split;
- the final prediction exposes feature and source-family contributions;
- raw sources remain linked so a reviewer can audit the evidence.

An LLM can later be added as an explanation layer, but it should not be the unobservable source of the numeric forecast.

## Why ridge regression for the live fusion model

The design goal is a coherent, inspectable AI product rather than architectural novelty. Ridge is useful here because it is:

- fast enough to refit per query;
- deterministic;
- stable with correlated technical features;
- easy to validate and unit-test;
- straightforward to explain through standardized feature contributions.

A more complex model should replace it only after a frozen empirical comparison shows material out-of-sample benefit.

## Why fitted weights use historical news but never Reddit content

Reddit's developer/data terms impose explicit restrictions on training algorithmic/AI models with Reddit content without appropriate permissions. The implementation therefore keeps Reddit completely outside model fitting: no historical Reddit feature enters the ridge training matrix and no coefficient is learned from Reddit content.

Current Reddit evidence can still influence the research forecast through a deliberately small, fixed and bounded inference-time overlay. That makes the effect inspectable while avoiding a hidden Reddit-trained model. The same internal interface supports a future licensed social-data feed if commercial rights are obtained.


## Why validation is purged

A future-return target spans the selected forecast horizon. Without a gap, the final training targets can overlap the dates used by the chronological holdout. The live model therefore removes a purge block equal to the horizon before validation begins. Holdout size and an approximate non-overlapping observation count are returned so compact-history results are not presented with false precision.

## Why ridge alpha is selected rather than fixed

The serving model evaluates a small declared alpha grid with generalized cross-validation on the outer training block only. The holdout remains untouched by this selection. This removes the unexplained fixed `0.35` penalty while preserving a lightweight deterministic fit when free-tier history is small.

## Why the Reddit overlay remains fixed

Reddit content is intentionally excluded from fitted model weights. The overlay cap of `0.25 × holdout residual sigma` is a design guardrail, not an empirically optimal coefficient: it limits social evidence to a secondary, inspectable adjustment and prevents it from dominating the fitted price+news signal.

## Why no Reddit HTML scraping

Scraping would create brittle selectors, unclear access rights and anti-bot workarounds. The live path therefore requires authorised API access and reports when Reddit is unavailable.

## Why Alpha Vantage rather than an undocumented finance endpoint

Data provenance is a product requirement. Alpha Vantage documents symbol search, daily time series and news/sentiment APIs. The provider adapter can later be swapped for Polygon, Finnhub, Bloomberg, Refinitiv or another licensed source.

## Why keep the offline fixture path

External APIs introduce non-determinism, rate limits, credentials and availability failures. A serious repository should still be testable after cloning.

The fixture path therefore remains first-class and covers:

- artifact loading;
- inference contracts;
- backtesting accounting;
- walk-forward evaluation;
- API/auth/cache behaviour;
- frontend contracts.

Live adapters are tested with mocked provider responses.

## Why TypeScript without a heavy frontend framework

The dashboard is intentionally dependency-light and strict-TypeScript. This keeps a clean checkout reproducible in constrained environments and makes the API/product work the focus. The backend contracts are framework-agnostic and can be moved into React/Next.js if component/state complexity justifies it.

## Why not claim “all of Reddit”

The conventional Reddit API supports post search but not a global arbitrary comment-search firehose. The product searches matched public posts, paginates within a configured budget, then retrieves top comments from the most engaged matched threads. The UI tells the user exactly what was covered.

Claiming more would be misleading.

## Why source-level graceful degradation

Alternative data are noisy and external. A news timeout should not erase a valid price-history analysis, and an unavailable Reddit token should not become a 500.

Each source therefore fails independently; the result records source errors and evidence counts so missing information is visible to the model consumer.

## What I would change for a production trading system

This repository is a research platform. A regulated or capital-bearing system would additionally need:

- licensed point-in-time data and clear entitlements;
- market-calendar/session handling;
- corporate actions, delistings and survivorship policy;
- immutable feature snapshots;
- experiment/model lineage;
- approval and rollback workflow;
- monitoring for drift and data quality;
- audit logging;
- stronger secrets management;
- portfolio-level risk constraints;
- execution venue integration separated from forecasting;
- compliance/legal review of alternative-data rights;
- kill switches and human approval where appropriate.

## Free-tier API budget engineering

The live path treats external API quota as a product constraint, not an afterthought. The browser never launches a live analysis automatically. Symbol resolution, history, news and Reddit results are cached independently beneath the final forecast cache, so changing model horizon does not trigger redundant upstream calls.

Alpha Vantage calls are serialized and paced. Daily history defaults to the provider's compact mode, which supplies enough recent observations for this lightweight live model while avoiding the premium-only full-history setting. Explicit uppercase ticker input can also bypass symbol lookup.

This layered cache design is intentional: caching only the final forecast would still waste quota whenever a user changed `5d` to `20d`, because the underlying market/news data are identical.
