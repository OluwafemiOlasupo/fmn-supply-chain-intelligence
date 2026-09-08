# Data Quality Report — project1_supply_chain_demand.csv

Independently validated against the raw CSV (not taken on faith from the
handoff brief) — see notebooks/ for the cell-by-cell validation trail.

## Confirmed as accurate
- Shape: 4,551 rows x 7 columns; 28 SKUs, 180 distinct dates, Jan 1 - Jun 29 2026
- 15 exact duplicate rows (each a single date+SKU double-entry, spread across 15 SKUs)
- 90 missing `units_sold`, 46 missing `closing_stock`
- Category label casing variants (e.g. "SNACKS" vs "Snacks") - fixed via `.str.title()`
- New SKUs SKU-2000/2001/2002: exactly 12 records each, Jun 18-29
- Total demand (~1.32M units), category demand shares, monthly totals, ABC tier split
  (16 SKUs ~= 79.6% of demand), high-CV SKU ranking, inventory-flow exact-match count (4,061),
  low-stock proxy rate (~36.6%) all reproduced exactly.

## Corrected — the handoff brief's EDA had an error
**Stockout day counts were wrong.** The original EDA counted missing
`closing_stock` values as stockouts (`closing_stock == 0 OR NaN`), which
contradicts its own stated principle of excluding missing values from
aggregates. This was proven by reproducing the original (wrong) figures
exactly under that assumption.

| | Original (incorrect) | Corrected |
|---|---|---|
| Total stockout SKU-days | 320 (7.1%) | **275 (6.1%)** |
| Snacks | 108 | **98** |
| Beverages | 89 | **77** |
| Sugar | (not given) | **69** |
| Pasta | (not given) | **16** |
| Flour | (not given) | **15** |

The directional story (Snacks/Beverages worst, stockouts trending up
toward June) is unchanged — only the magnitudes needed correcting. All
stockout calculations in this codebase use `src.data.load_clean.stockout_mask()`,
which enforces the corrected definition.

## New finding — cold-start SKUs are a distinct data pattern, not just "short history"
SKU-2000/2001/2002 show a materially different pattern from "established
SKU with less data":
- `units_received` = 0 for all 36 combined records — never restocked in
  the observed window.
- `closing_stock` decays to exactly 0 and stays there, while `units_sold`
  continues to post positive values on zero-stock days — a flow-consistency
  violation on top of the stockout itself.
- `lead_time_days` varies day-to-day for these 3 SKUs (4-5 distinct values
  each), whereas every established SKU has exactly one constant lead time.
  This looks like unconfirmed/placeholder supplier data for unlaunched
  replenishment lines.

Practical implication: applying the naive low-stock formula (or any
model fit on the SKU's own history) to these SKUs produces a misleading
"most urgent SKU in the dataset" signal that is really just an artifact
of zero stock + no restocking data. This is why cold-start SKUs are
scored with a completely separate path (`src/risk/scoring.py:score_cold_start_skus`)
using a category-level demand proxy and a hardcoded Low confidence flag,
rather than being run through the same model as established SKUs.

## Inventory-flow mismatches
269 of 4,330 comparable records (6.2%) show `closing_stock[t] !=
closing_stock[t-1] + units_received[t] - units_sold[t]`, with a mean
mismatch magnitude of ~208 units (max 640). These are surfaced, not
silently dropped: they feed into the confidence score for each SKU
(>15% mismatch rate in a SKU's history -> Medium confidence instead of High).
Root cause was not further investigated (deferred — plausible causes
include timing/backdating of receipts, unrecorded transfers, or data
entry issues; not established as any one cause).
