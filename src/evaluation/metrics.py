"""
Forecast evaluation metrics, and the chronological (non-random)
validation split used throughout this project. See docs/experiment_log.md
for the actual baseline-vs-GBM comparison results produced with these.
"""

import pandas as pd


def chronological_split(df: pd.DataFrame, date_col: str = "date", holdout_days: int = 30):
    """Time-aware split: everything in the last `holdout_days` is the
    test set. NEVER use a random split for this project's time-series
    forecasting task — see brief section 9."""
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff]
    test = df[df[date_col] > cutoff]
    return train, test, cutoff


def mae(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, int]:
    mask = y_true.notna() & y_pred.notna()
    return float((y_true[mask] - y_pred[mask]).abs().mean()), int(mask.sum())


def wape(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, int]:
    mask = y_true.notna() & y_pred.notna()
    total = y_true[mask].sum()
    return float((y_true[mask] - y_pred[mask]).abs().sum() / total), int(mask.sum())


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    m, n_mae = mae(y_true, y_pred)
    w, n_wape = wape(y_true, y_pred)
    return {"mae": m, "wape": w, "n": n_mae}
