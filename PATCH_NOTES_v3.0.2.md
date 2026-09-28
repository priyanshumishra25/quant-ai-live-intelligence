# v3.0.2 Modelling and cleanup patch

This release addresses temporal-validation, uncertainty, semantics, timestamp alignment and repository-scope issues found during review.

## Modelling correctness

- Purges `horizon` rows between training and chronological holdout.
- Keeps the outer holdout out of ridge-alpha selection.
- Selects ridge alpha from a declared grid using training-only generalized cross-validation.
- Derives forecast residual scale from holdout errors, not final-fit in-sample errors.
- Reports holdout rows, purge rows, approximate non-overlapping observations and a validation-reliability label.

## Output semantics

- Removes live BUY/HOLD/SELL pseudo-probabilities.
- Replaces them with `signal_score`, `signal_strength`, `bullish_score`, `neutral_score` and `bearish_score`.
- Separates source coverage from validation metrics.

## Temporal evidence

- Converts news timestamps to `America/New_York` for daily US-market alignment.
- News at/after 16:00 ET rolls to the next available market session.
- Weekend/holiday news rolls forward.
- Missing or malformed provider timestamps are dropped rather than replaced with `now()`.

## Repository scope

Disconnected prototype stacks were removed from the public mainline: RL, GNN/anomaly, Kafka streaming, scraping/proxy rotation, options, tree models, retraining, risk, legacy DB/ingestion and heavy research-only model scaffolding.

## Security and presentation

- Staging/production startup rejects known demo/default auth secrets.
- Kubernetes manifests contain placeholders rather than deployable demo credentials.
- Interview-specific repository language was removed.
- `CITATION.cff` now includes author metadata.

A software licence remains an explicit repository-owner choice and is not selected automatically by this patch.
