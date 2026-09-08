"""
Data loading and cleaning for the FMN Supply Chain Intelligence project.

All logic here was independently validated cell-by-cell against the raw
CSV before being ported into this module. See notebooks/ for the
validation trail, and README.md for a summary of what was checked.

Key corrections vs. the original EDA handed off for this task:
- Stockout counts must exclude missing closing_stock values, not treat
  them as zero. The original EDA's stockout figures (e.g. "7.1% of
  observations") were reproduced exactly by wrongly counting missing
  values as stockouts; see docs/data_quality_report.md for the full
  writeup and corrected figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd

NEW_SKUS = ["SKU-2000", "SKU-2001", "SKU-2002"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw supply chain CSV with dates parsed."""
    return pd.read_csv(path, parse_dates=["date"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the validated cleaning pipeline:
      - drop exact (date, sku_id) duplicates, keep first occurrence
      - normalize category labels (casing variants -> Title Case)
      - flag cold-start SKUs

    Does NOT impute missing units_sold / closing_stock — those are left
    as NaN and must be handled explicitly (and separately) by any
    downstream aggregation, per the data-quality principle of not
    silently treating missing as zero.
    """
    before = len(df)
    out = df.drop_duplicates(subset=["date", "sku_id"], keep="first").copy()
    dropped = before - len(out)
    if dropped:
        # Expected: 15 exact duplicate rows in the reference dataset.
        pass

    out["category"] = out["category"].str.strip().str.title()
    out["is_new_sku"] = out["sku_id"].isin(NEW_SKUS)

    return out


def stockout_mask(df: pd.DataFrame) -> pd.Series:
    """
    Correct stockout definition: closing_stock == 0, with missing
    values EXCLUDED (not treated as stockouts). Use this everywhere
    stockout counts are computed, to avoid reproducing the original
    EDA's inflated figures.
    """
    return df["closing_stock"] == 0


def load_and_clean(path: str | Path) -> pd.DataFrame:
    return clean(load_raw(path))
