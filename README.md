# FMN Supply Chain Intelligence

A decision-support system for Flour Mills of Nigeria's supply chain team: identifies
which SKUs need attention (stockout risk) over their replenishment lead time, explains
why in plain English, and answers free-text questions — all grounded in the actual data.

## The story, in one paragraph

FMN has uneven inventory outcomes: some SKUs stock out unexpectedly while others sit
overstocked. This system forecasts demand over each SKU's replenishment lead time using
a weekday-aware seasonal baseline (chosen over a gradient-boosted model after rigorous,
honest comparison — see `docs/experiment_log.md`), compares that forecast to current
stock, and separates **risk** (how urgent) from **confidence** (how much to trust the
number) — a new SKU can be high risk with low confidence; an established SKU can be high
risk with high confidence. An LLM (DeepSeek) sits strictly downstream of this analytical
layer: it only explains evidence that's already been computed, never decides risk itself
and never invents numbers.

## Quick start (Local Run)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed — already gitignored):
```
DEEPSEEK_API_KEY=your_key_here
```

Place the dataset at `data/raw/project1_supply_chain_demand.csv`, then run:
```bash
streamlit run app/app.py
```

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Project structure

```
src/data/       — loading, cleaning (dedup, category normalization, cold-start flagging)
src/features/   — baseline forecasts (winner) + GBM feature set (documented, rejected)
src/risk/       — risk/decision layer: forecast -> shortfall -> tier + confidence
src/llm/        — evidence-object construction, DeepSeek explanations, grounded Q&A
src/evaluation/ — metrics (MAE/WAPE) and the chronological validation split
src/pipeline.py — single entry point: raw CSV -> master risk table
app/app.py      — Streamlit dashboard
tests/          — unit tests, including deliberately-chosen edge cases
docs/           — data quality report, experiment log
notebooks/      — exploratory validation trail: Step 1 data-quality checks (including the
                  stockout-count correction), baseline vs. GBM comparison, run cell-by-cell
                  against the raw dataset before being ported into src/
```

## Key decisions and why

**Target formulation.** Predict demand over each SKU's `lead_time_days` (its actual
replenishment horizon), not a single day — a one-day proxy is noisy and, as shown in
testing, actively misleading for new SKUs. Coverage ratio (`current_stock /
forecasted_demand_over_horizon`) drives a 4-tier risk label: High (<0.7), Medium
(0.7–1.0), Low (1.0–3.0), Overstock (>3.0), plus an `Unknown` fallback for the edge case
where forecasted demand is exactly zero (see `docs/experiment_log.md` / tests).

**Model selection.** A weekday-aware seasonal baseline (per-SKU, per-weekday average
demand) beat a gradient-boosted regressor on the same chronological holdout, even after
feature enrichment and regularization — the dataset (≈124 rows/SKU) doesn't have enough
signal for a tree model to out-learn a simple seasonal average. We use the baseline in
production and document the rejected experiment rather than hiding it. No `.pkl` model
artifact exists because the "model" is a lookup table recomputed at scoring time, not a
fitted estimator.

**Validation.** Chronological holdout only (last 30 days) — never a random split, since
this is a time-series forecasting problem and a random split would leak future
information into training.

**New / cold-start SKUs.** Never scored with their own history (12 days is too little,
and in this dataset they also show zero replenishment and unstable lead-time data — a
real, deeper anomaly, not just short history). They get a category-level demand proxy
and a hardcoded Low-confidence flag instead.

**LLM architecture.** The LLM (DeepSeek) receives only a precomputed, structured evidence
dict per SKU (`src/llm/evidence.py`) — it never touches raw data and is explicitly
instructed not to override the risk tier or confidence already assigned. Explanations are
generated on-demand (button click in the app), not eagerly for every SKU on page load.
Grounded Q&A uses a deterministic intent-classification + retrieval step before any
natural-language answer is generated — no vector RAG, per the small structured dataset.

## Assumptions

- Exact duplicate `(date, sku_id)` rows are accidental double-entries; first occurrence
  kept, not averaged or summed.
- Missing `units_sold` / `closing_stock` are treated as missing, never imputed as zero,
  anywhere in the pipeline — this is the exact mistake that produced the original EDA's
  incorrect stockout figures.
- `lead_time_days` is treated as each established SKU's fixed replenishment horizon
  (it is fixed in the data for all established SKUs). For new SKUs it is not fixed in the
  data, so only the most recently recorded value is used, alongside the hardcoded
  Low-confidence flag.
- Inventory-flow mismatches are surfaced as a confidence signal, not silently corrected —
  root cause is unknown, so no fix is assumed.

## Deployment URL: 
https://fmn-supply-chain-intelligence-fbrhcz6onmwgyqwbpxhbha.streamlit.app/

## Known limitations

- Root cause of the 269 inventory-flow mismatches not further investigated (deferred;
  plausible causes include timing/backdating of receipts or unrecorded transfers — not
  established as any one cause).
- Weekly seasonality (Wednesday high / Saturday low) is based on ~6 months of data;
  should be revalidated as more history accumulates.
- Cold-start category-average proxy is a reasonable fallback, not a tuned model — more
  new-SKU history, launch plans, or comparable-SKU matching would improve it.
