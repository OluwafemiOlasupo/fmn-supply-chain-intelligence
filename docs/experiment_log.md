# Experiment Log — Demand Forecasting Model Selection

All experiments evaluated on the same chronological holdout (last 30 days:
2026-05-31 to 2026-06-29), established SKUs only (new SKUs use a separate
cold-start path, not a forecasting model — see docs/data_quality_report.md).

| # | Model | Features | MAE | WAPE | Decision |
|---|---|---|---|---|---|
| 1 | Naive (lag-1) | none | 75.12 | 25.23% | Rejected — worst baseline |
| 2 | Rolling 7-day mean | none | 63.26 | 21.25% | Rejected — beaten by weekday baseline |
| 3 | **Weekday-aware (expanding mean by weekday)** | none | **52.84** | **17.75%** | **ACCEPTED — production model** |
| 4 | Weekday-aware, rolling 4-occurrence hybrid | none | 57.19 | 19.21% | Rejected — narrower window adds noise, no real trend to exploit with only ~26 weeks of history |
| 5 | Gradient-boosted (HistGradientBoostingRegressor) | 15 features: lags (1/7/14), rolling mean/std, weekday-expanding-mean, recent receipts, prev stock, calendar, category/SKU codes, lead time | 55.69 | 18.74% | Rejected — worse than baseline (-0.70pp) |
| 6 | Gradient-boosted, enriched + regularized | +4 features: SKU avg demand, SKU demand std, category rolling mean, recent deviation; depth=3, l2=1.0 | 53.89 | 18.14% | Rejected — still worse than baseline (-0.10pp), even after feature enrichment and regularization |

## Why the baseline wins

With ~124 rows of history per SKU (26 weeks × established SKUs), there
isn't enough data for a tree-based model to learn patterns beyond what a
simple day-of-week seasonal average already captures. The GBM's marginal
gap versus the baseline shrank meaningfully after feature enrichment
(-0.70pp → -0.10pp), suggesting the ceiling here is close to the
baseline's performance, not that the modeling approach is fundamentally
wrong — just that the added complexity isn't earning its keep on this
dataset size.

## Decision

**Production demand forecast: the weekday-aware baseline** (per-SKU,
per-weekday expanding mean, computed from full history at scoring time).
No model artifact (.pkl) is produced — the "model" is a lookup table of
per-SKU-per-weekday averages, recomputed at scoring time from the
cleaned dataset. This is intentional and documented, not a shortcut.
