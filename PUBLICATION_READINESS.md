# Portfolio / Publication Readiness

## What v3.0.2 is ready for

The repository is suitable as a public software + AI engineering project after the owner confirms the public Git history contains no secrets and completes the final release checks below.

Repository-controlled readiness work is implemented:

- live documented-provider integration;
- arbitrary stock/company resolution;
- market-session-aware news alignment;
- purged chronological validation;
- training-only ridge-alpha selection;
- holdout-residual uncertainty;
- heuristic signal scores labelled as scores rather than probabilities;
- explicit small-sample validation reliability;
- source-attributed model output;
- provider degradation handling;
- deterministic CI without live secrets;
- production startup rejection of known demo/default authentication secrets;
- API, frontend, cache, auth and observability concerns.

## What it must not claim

Do not claim that the current live fusion model has demonstrated durable real-market alpha.

A scientific/performance claim requires a separately frozen empirical protocol including:

- licensed point-in-time data;
- corporate actions / ticker changes / delistings;
- survivorship-bias policy;
- immutable historical news/social snapshots;
- exchange-specific market calendars and timestamp policy;
- untouched final test set;
- hyperparameter-selection disclosure;
- baseline + ablation studies;
- repeated experiments where applicable;
- transaction-cost and slippage sensitivity;
- statistical uncertainty / multiple-testing controls;
- social-data usage rights appropriate to the study.

## Repository-controlled checks

- [x] `CITATION.cff` contains author metadata.
- [x] `.env` is excluded from the packaged release and release checks reject tracked secrets.
- [x] Tests isolate live provider credentials.
- [x] Staging/production startup rejects the known demo password and placeholder JWT secrets.
- [x] Kubernetes secret values are placeholders rather than deployable demo credentials.
- [x] Live signal outputs are explicitly scores, not calibrated probabilities.
- [x] Validation uses a horizon-length purge gap.
- [x] Forecast intervals use holdout residuals.
- [x] News after the US close rolls to the next market session.
- [x] Invalid news timestamps are dropped rather than fabricated.

## Owner actions before public release

- [x] Add MIT software licence.
- [x] Inspect Git history for any previously committed provider secrets and rotate any exposed credentials.
- [x] Run `make verify` from a clean checkout.
- [x] Build both Docker images in CI.
- [x] `CITATION.cff` identifies the author, repository, release version and MIT licence.
- [ ] Add screenshots/GIF if desired.
- [ ] If hosting a demo, inject provider keys and authentication secrets only server-side.
- [ ] Confirm provider terms permit the intended hosted use.
