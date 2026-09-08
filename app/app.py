"""
FMN Supply Chain Intelligence — Streamlit dashboard.

Business-facing app for operations staff: see which SKUs need
attention, why, and ask free-text questions grounded in the data.

Run locally:
    streamlit run app/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

# Allow running as `streamlit run app/app.py` from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.evidence import build_evidence
from src.llm.explain import explain_sku, get_client
from src.llm.qa import answer_question
from src.pipeline import run_pipeline

st.set_page_config(page_title="FMN Supply Chain Intelligence", layout="wide")

DATA_PATH = os.environ.get("SUPPLY_CHAIN_CSV_PATH", "data/raw/project1_supply_chain_demand.csv")


@st.cache_data(show_spinner="Loading and scoring supply chain data...")
def load_data(path: str):
    return run_pipeline(path)


def risk_color(tier: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Overstock": "🔵", "Unknown": "⚫"}.get(tier, "⚪")


def main():
    st.title("📦 FMN Supply Chain Intelligence")
    st.caption(
        "Identifies SKUs at risk of stocking out (or overstocked) over their replenishment "
        "lead time, and explains why — grounded in actual demand and inventory data."
    )

    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`. Set the SUPPLY_CHAIN_CSV_PATH environment "
            "variable, or place the CSV at that path."
        )
        st.stop()

    df_clean, master_risk_full, evidence_supplement, est_df = load_data(DATA_PATH)

    # ---- KPI row -----------------------------------------------------
    n_high = (master_risk_full["risk_tier"] == "High").sum()
    n_medium = (master_risk_full["risk_tier"] == "Medium").sum()
    n_low_confidence = (master_risk_full["confidence"] != "High").sum()
    n_new_skus = master_risk_full["is_new_sku"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 High risk SKUs", int(n_high))
    c2.metric("🟡 Medium risk SKUs", int(n_medium))
    c3.metric("⚠️ Reduced-confidence flags", int(n_low_confidence))
    c4.metric("🆕 New / cold-start SKUs", int(n_new_skus))

    st.divider()

    # ---- Prioritized risk table with filters --------------------------
    st.subheader("SKU Risk Overview")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        tier_filter = st.multiselect(
            "Risk tier", options=["High", "Medium", "Low", "Overstock", "Unknown"],
            default=["High", "Medium"],
        )
    with fcol2:
        category_filter = st.multiselect(
            "Category", options=sorted(master_risk_full["category"].unique()), default=[]
        )
    with fcol3:
        confidence_filter = st.multiselect(
            "Confidence", options=["High", "Medium", "Low"], default=[]
        )

    filtered = master_risk_full.copy()
    if tier_filter:
        filtered = filtered[filtered["risk_tier"].isin(tier_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if confidence_filter:
        filtered = filtered[filtered["confidence"].isin(confidence_filter)]

    filtered = filtered.sort_values(
        "risk_tier", key=lambda c: c.map({"High": 0, "Medium": 1, "Low": 2, "Overstock": 3, "Unknown": 4})
    )

    display_df = filtered[[
        "sku_id", "category", "risk_tier", "confidence", "current_stock",
        "forecast_demand_horizon", "coverage_ratio", "lead_time_days", "is_new_sku",
    ]].copy()
    display_df["risk_tier"] = display_df["risk_tier"].apply(lambda t: f"{risk_color(t)} {t}")
    display_df.columns = [
        "SKU", "Category", "Risk", "Confidence", "Current Stock",
        "Forecast Demand (horizon)", "Coverage Ratio", "Lead Time (days)", "New SKU",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- SKU detail view -----------------------------------------------
    st.subheader("SKU Detail")
    selected_sku = st.selectbox("Select a SKU to inspect", options=master_risk_full["sku_id"].sort_values())

    sku_row = master_risk_full[master_risk_full["sku_id"] == selected_sku].iloc[0]
    ev = build_evidence(sku_row)

    dcol1, dcol2 = st.columns([1, 1])
    with dcol1:
        st.markdown(f"### {selected_sku} — {risk_color(sku_row['risk_tier'])} {sku_row['risk_tier']} risk")
        st.markdown(f"**Confidence:** {sku_row['confidence']} — _{sku_row['confidence_reason']}_")
        if sku_row["is_new_sku"]:
            st.warning(
                "This is a new SKU (< 2 weeks of history, no replenishment observed, unstable "
                "lead-time data). Its forecast uses a category-level proxy, not its own history — "
                "treat these numbers with extra caution."
            )
        st.write(f"**Current stock:** {ev['current_stock_units']:.0f} units")
        st.write(f"**Lead time:** {ev['lead_time_days']} days")
        st.write(f"**Forecasted demand over lead time:** {ev['forecasted_demand_over_lead_time']:.0f} units")
        st.write(f"**Shortfall / surplus:** {ev['shortfall_or_surplus_units']:.0f} units")
        st.write(f"**Coverage ratio:** {ev['coverage_ratio']:.2f}")
        if not sku_row["is_new_sku"]:
            st.write(f"**ABC tier:** {ev.get('abc_tier', 'N/A')}")
            st.write(f"**Demand trend (last 30d vs prior 30d):** {ev.get('demand_trend_last_30d_pct')}%")
            st.write(f"**Demand volatility (CV):** {ev.get('demand_volatility_cv')}")
            st.write(f"**Historical stockout days (6mo):** {ev.get('historical_stockout_days_last_6mo')}")

    with dcol2:
        sku_history = df_clean[df_clean["sku_id"] == selected_sku].sort_values("date")
        if not sku_history.empty:
            st.markdown("**Demand & inventory history**")
            chart_df = sku_history.set_index("date")[["units_sold", "closing_stock"]]
            st.line_chart(chart_df)
            st.markdown("**Receipts (units received)**")
            st.bar_chart(sku_history.set_index("date")[["units_received"]])

    st.markdown("---")
    if st.button("🧠 Click here to get detailed SKU explanation", key=f"explain_{selected_sku}"):
        try:
            with st.spinner("Generating grounded explanation..."):
                client = get_client()
                explanation = explain_sku(ev, client)
            st.info(explanation)
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not generate explanation: {e}")

    st.divider()

    # ---- Grounded Q&A -----------------------------------------------
    st.subheader("💬 Ask a question about the data")
    question = st.text_input(
        "e.g. \"Which SKUs are at highest risk?\", \"Why is SKU-1017 flagged?\", "
        "\"Which category has the most stockouts?\""
    )
    if st.button("Ask") and question.strip():
        try:
            with st.spinner("Looking into it..."):
                client = get_client()
                answer = answer_question(question, master_risk_full, est_df, evidence_supplement, client)
            st.success(answer)
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not answer question: {e}")


if __name__ == "__main__":
    main()
