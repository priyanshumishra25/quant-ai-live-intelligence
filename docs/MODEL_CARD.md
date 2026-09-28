# Model Card — Live Fusion Forecast

## Model

`on_demand_ridge_fusion`

## Purpose

Produce an inspectable research forecast for a requested listed instrument using a fitted historical price+news model, with current Reddit discussion available only as a fixed, bounded inference-time overlay.

## Intended use

- exploratory market intelligence;
- evaluation of data/model-serving architecture;
- reproducible analysis of temporal feature engineering and validation behaviour;
- inspection of source attribution, uncertainty and provider degradation.

## Not intended for

- autonomous trade execution;
- personalized financial advice;
- claims of guaranteed or persistent alpha;
- high-frequency/intraday execution;
- training persistent models on Reddit user content.

## Inputs

Technical:

- 1-day return;
- 5-day return;
- 5-day MA gap;
- 20-day MA gap;
- 10-day annualised volatility;
- 10-day momentum.

Fitted alternative-data inputs:

- news sentiment;
- news volume.

Inference-only social inputs:

- current Reddit sentiment;
- current Reddit volume;
- current Reddit engagement.

## Target

Future close-to-close return over 1, 5 or 20 trading days.

## News timing policy

Historical news is aligned to the first US market close at which it is knowable. Timestamps are converted to `America/New_York`. News published before 16:00 ET may enter that session's close feature; news at or after 16:00 ET rolls to the next available market session. Weekend and holiday items also roll forward. Items with missing or malformed timestamps are dropped rather than assigned a fabricated time.

## Training and regularisation

Model weights are fit on demand from an historical matrix containing technical and news features using L2-regularised linear regression. Features are standardized using the training block's mean/std. Reddit-derived signals are never included in this fitting matrix.

The ridge penalty is **not a fixed magic constant**. Alpha is selected from a small declared grid using generalized cross-validation on the outer training block only. The chronological holdout is never consulted during hyperparameter selection.

## Validation

The most recent chronological segment is reserved as a holdout. A purge gap equal to the forecast horizon is removed between the training block and holdout so training targets cannot extend into the validation period.

The API reports:

- holdout row count;
- purge-row count;
- MAE;
- directional accuracy;
- prediction/target correlation;
- an approximate count of non-overlapping validation observations;
- `LOW` / `MEDIUM` / `HIGH` validation reliability.

With compact free-tier history, the 5-day and especially 20-day holdouts can contain very few effectively independent observations. Directional accuracy and correlation are therefore diagnostics, not evidence of a tradable edge.

## Inference and uncertainty

After holdout evaluation, the selected ridge penalty is used for a final refit on all eligible historical observations. The current technical + news vector is then scored.

The reported forecast interval uses the **purged holdout residual standard deviation**, not residuals from the final in-sample refit. This is still a lightweight residual interval rather than a fully calibrated predictive distribution, and its reliability is explicitly limited when the holdout is small.

## Signal scores — not probabilities

The live API does not label the heuristic BUY/HOLD/SELL softmax as probability. Instead it reports:

```text
signal_score      signed value in [-1, 1]
signal_strength   abs(signal_score)
bullish_score
neutral_score
bearish_score
```

The three directional scores form a normalized diagnostic decomposition, but they are **not calibrated class probabilities** and must not be interpreted as probabilities of profit.

## Reddit inference boundary

Current Reddit sentiment may affect the forecast only through a fixed bounded inference-time overlay. The maximum magnitude is capped at **0.25 holdout-residual sigma** and scaled by current Reddit evidence strength.

The `0.25` value is a **design guardrail, not an empirically optimized coefficient**: it deliberately prevents noisy social evidence from dominating the fitted price+news model while keeping the effect visible and inspectable. No model coefficient is learned from Reddit content.

## Explainability

The API exposes standardized coefficient contributions and groups their absolute effect into:

```text
technical
news
Reddit
```

This is local linear contribution, not causal attribution.

## Source coverage

Source coverage is a separate evidence-volume diagnostic. Validation accuracy is not mixed into this value, and source coverage does not turn the signal score into a probability or confidence estimate.

## Data / rights boundary

Reddit content is accessed only through authorised API credentials and processed ephemerally. It is not inserted into the historical training matrix, used to fit model weights, or stored as a training corpus/model artefact. Provider terms and rights must be re-evaluated before any commercial deployment.

## Known failure modes

- sparse news/social coverage;
- ambiguous company names;
- ticker changes / corporate actions;
- coordinated social activity;
- sentiment sarcasm/slang;
- regime shifts;
- feature collinearity;
- provider outages/rate limits;
- limited free-tier history and low effective holdout sample size;
- exchange/session assumptions for non-US listings;
- weak statistical power for less-liquid/newly listed securities.

## Production path

A production-grade successor should use licensed point-in-time data, exchange-specific calendars, immutable feature snapshots, experiment/model lineage, stronger NLP evaluation, explicit corporate-action policy, a frozen untouched test set, drift monitoring and portfolio-level risk controls.
