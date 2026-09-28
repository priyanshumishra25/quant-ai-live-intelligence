# Model Card — Live Fusion Forecast

## Model

`on_demand_ridge_fusion`

## Purpose

Produce an inspectable research forecast for a requested listed instrument using a fitted historical price+news model, with current Reddit discussion available only as a fixed, bounded inference-time overlay.

## Intended use

- software/AI engineering demonstration;
- exploratory market research;
- evaluation of data/model-serving architecture;
- interview discussion of feature leakage, provider reliability, alternative data and model validation.

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

## Training

Model weights are fit on demand from an historical matrix containing technical and news features using L2-regularised linear regression. Features are standardized using training-history mean/std. Reddit-derived signals are never included in this fitting matrix.

## Validation

The most recent chronological segment is withheld before the final refit. The API reports holdout MAE, directional accuracy and correlation.

## Inference

The latest technical vector and seven-day news state are scored by the fitted ridge model. Current Reddit sentiment, when available, is then applied through a fixed bounded overlay whose magnitude is scaled by Reddit evidence strength; no Reddit coefficient is learned. Predictions are clipped to horizon-specific sanity bounds before presentation. Uncertainty uses the fitted model residual standard deviation.

## Explainability

The API exposes standardized coefficient contributions and groups their absolute effect into:

```text
technical
news
Reddit
```

This is local linear contribution, not causal attribution.

## Evidence quality

A separate score combines source coverage with validation diagnostics. It intentionally reduces displayed confidence when evidence is sparse or holdout directionality is weak.

It must not be interpreted as probability of profit.

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
- timezone edge cases;
- weak statistical power for less-liquid/newly listed securities.

## Production path

A production-grade successor should use licensed point-in-time data, immutable feature snapshots, experiment/model lineage, stronger NLP evaluation, explicit corporate-action policy, a frozen untouched test set, drift monitoring and portfolio-level risk controls.
