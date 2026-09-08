"""
End-to-end pipeline: raw CSV -> cleaned data -> risk table + evidence
supplement, ready for the Streamlit app and LLM layer.

Usage:
    from src.pipeline import run_pipeline
    df_clean, master_risk, evidence_supplement, est_df = run_pipeline("data/raw/project1_supply_chain_demand.csv")
"""

import pandas as pd

from src.data.load_clean import load_and_clean
from src.llm.evidence import compute_evidence_supplement
from src.risk.scoring import build_master_risk_table


def run_pipeline(csv_path: str):
    df_clean = load_and_clean(csv_path)

    est_df = df_clean[~df_clean["is_new_sku"]].copy()
    est_df["weekday"] = est_df["date"].dt.dayofweek

    master_risk = build_master_risk_table(df_clean)
    evidence_supplement = compute_evidence_supplement(est_df)

    master_risk_full = master_risk.merge(evidence_supplement, on="sku_id", how="left")

    return df_clean, master_risk_full, evidence_supplement, est_df


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/project1_supply_chain_demand.csv"
    df_clean, master_risk_full, evidence_supplement, est_df = run_pipeline(path)
    print(f"Cleaned dataset: {df_clean.shape}")
    print(f"Master risk table: {master_risk_full.shape}")
    print(master_risk_full["risk_tier"].value_counts())
