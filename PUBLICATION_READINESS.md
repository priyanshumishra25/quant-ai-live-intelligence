# Portfolio / Publication Readiness

## What v3.0.1 is ready for

The repository is suitable as a public **software + AI engineering portfolio project** once the author selects a licence and replaces placeholder author metadata.

It demonstrates:

- live documented-provider integration;
- arbitrary stock/company resolution;
- alternative-data feature engineering;
- chronological ML validation;
- source-attributed model output;
- provider degradation handling;
- deterministic CI without live secrets;
- API, frontend, cache, auth and observability concerns.

## What it must not claim

Do not claim that the current live fusion model has demonstrated durable real-market alpha.

A scientific/performance claim requires a separately frozen empirical protocol including:

- licensed point-in-time data;
- corporate actions / ticker changes / delistings;
- survivorship-bias policy;
- immutable historical news/social snapshots;
- strict market-session/timezone policy;
- untouched final holdout;
- hyperparameter-selection disclosure;
- baseline + ablation studies;
- repeated experiments where applicable;
- transaction-cost and slippage sensitivity;
- statistical uncertainty / multiple-testing controls;
- social-data usage rights appropriate to the study.

## Public repository checklist

- [ ] Choose and add a software licence.
- [ ] Replace placeholder author metadata in `CITATION.cff`.
- [ ] Confirm `.env` and provider secrets are absent from Git history.
- [ ] Run `make verify` from a clean checkout.
- [ ] Build both Docker images in CI.
- [ ] Add screenshots/GIF of Live Intelligence.
- [ ] If hosting a demo, inject provider keys only server-side.
- [ ] Ensure provider terms permit the intended hosted use.
- [ ] Change demo auth credentials and JWT secret.
- [ ] Add a short architecture diagram to the GitHub README preview.

## Interview demo sequence

A strong five-minute demo is:

1. Type a company name in **Live Intelligence**.
2. Show symbol resolution and historical trend.
3. Show news + Reddit evidence and source attribution.
4. Show the model's technical/news/Reddit contribution mix.
5. Show chronological validation rather than random train/test splitting.
6. Open **System** to discuss provider boundaries, cache, API and observability.
7. Open **Experiment Lab** to show baseline discipline.
8. Explain why the deterministic offline path exists for CI.
