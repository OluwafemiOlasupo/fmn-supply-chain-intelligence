"""
Grounded free-text Q&A over the supply-chain dataset.

Design (per brief guidance: prefer deterministic retrieval over vector
RAG for small structured tabular data):

  1. LLM classifies the free-text question into one of a FIXED set of
     known intents + extracts parameters (e.g. a SKU id). Returns JSON.
  2. A plain Python function retrieves the actual answer data — no LLM
     involved in this step, so numbers can't be hallucinated here.
  3. LLM phrases the final answer in natural language, using ONLY the
     data retrieved in step 2.

If the question doesn't match a known intent, we say so rather than
guessing — this is deliberately narrow rather than open-ended RAG.
"""

import json

import pandas as pd
from openai import OpenAI

from src.data.load_clean import stockout_mask
from src.llm.evidence import build_evidence
from src.llm.explain import DEEPSEEK_MODEL, get_client

QA_INTENTS_PROMPT = """You must classify the user's question into exactly one of these intents,
and extract any parameters needed. Respond with ONLY valid JSON, no other text.

Available intents:
- "highest_risk": which SKUs are highest risk. params: {"n": int, default 5}
- "sku_detail": details/explanation for a specific SKU. params: {"sku_id": string, e.g. "SKU-1017"}
- "stockouts_by_category": which category has most stockouts. params: {}
- "demand_trend": how demand has changed recently. params: {}
- "high_demand_high_variability": which high-demand SKUs are also volatile. params: {"n": int, default 5}
- "unknown": question doesn't match any of the above. params: {}

Respond in this exact JSON format: {"intent": "...", "params": {...}}
"""

UNKNOWN_INTENT_MESSAGE = (
    "I can answer questions about SKU risk, stockouts, demand trends, and volatility — "
    "could you rephrase?"
)


def classify_question(question: str, client: OpenAI) -> dict:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": QA_INTENTS_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=100,
    )
    return json.loads(response.choices[0].message.content.strip())


# --- Deterministic retrieval functions -------------------------------------
# Each takes the already-computed risk/evidence tables and returns plain
# data (dicts/lists), never a natural-language answer.

def get_highest_risk_skus(master_risk: pd.DataFrame, n: int = 5) -> list[dict]:
    cols = ["sku_id", "category", "risk_tier", "confidence", "coverage_ratio", "current_stock"]
    return (
        master_risk[master_risk["risk_tier"] == "High"]
        .sort_values("coverage_ratio")[cols]
        .head(n)
        .to_dict("records")
    )


def get_sku_detail(master_risk: pd.DataFrame, sku_id: str) -> dict:
    row = master_risk[master_risk["sku_id"] == sku_id]
    if row.empty:
        return {"error": f"{sku_id} not found"}
    return build_evidence(row.iloc[0])


def get_stockouts_by_category(est_df: pd.DataFrame) -> dict:
    """
    Scoped to ESTABLISHED SKUs only. New SKUs' zero-stock days reflect
    "never restocked since launch" (a cold-start data issue, see
    docs/data_quality_report.md), not a normal stockout, so they're
    intentionally excluded from this operational KPI. This differs from
    the whole-dataset stockout figure reported in the data-quality
    report (275), which deliberately includes new SKUs for data-quality
    purposes. Both numbers are correct for what they measure.
    """
    stockouts = est_df[stockout_mask(est_df)]
    return stockouts.groupby("category").size().sort_values(ascending=False).to_dict()


def get_demand_trend(evidence_supplement: pd.DataFrame) -> list[dict]:
    return evidence_supplement[["sku_id", "trend_pct"]].sort_values("trend_pct").to_dict("records")


def get_high_demand_high_variability_skus(evidence_supplement: pd.DataFrame, est_df: pd.DataFrame,
                                            n: int = 5) -> list[dict]:
    avg_demand = est_df.groupby("sku_id")["units_sold"].mean().rename("avg_demand")
    merged = evidence_supplement.merge(avg_demand, on="sku_id")
    return (
        merged.sort_values(["avg_demand", "cv"], ascending=[False, False])[["sku_id", "avg_demand", "cv"]]
        .head(n)
        .to_dict("records")
    )


def answer_question(question: str, master_risk: pd.DataFrame, est_df: pd.DataFrame,
                     evidence_supplement: pd.DataFrame, client: OpenAI | None = None) -> str:
    """Top-level entry point: classify -> retrieve -> phrase."""
    client = client or get_client()
    classification = classify_question(question, client)
    intent, params = classification.get("intent"), classification.get("params", {})

    if intent == "highest_risk":
        data = get_highest_risk_skus(master_risk, params.get("n", 5))
    elif intent == "sku_detail":
        data = get_sku_detail(master_risk, params.get("sku_id", ""))
    elif intent == "stockouts_by_category":
        data = get_stockouts_by_category(est_df)
    elif intent == "demand_trend":
        data = get_demand_trend(evidence_supplement)
    elif intent == "high_demand_high_variability":
        data = get_high_demand_high_variability_skus(evidence_supplement, est_df, params.get("n", 5))
    else:
        return UNKNOWN_INTENT_MESSAGE

    answer_prompt = f"""Answer this question using ONLY the data below. Be concise (2-4 sentences).
Do not invent numbers not present in the data.

Question: {question}

Data: {data}
"""
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.3,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()
