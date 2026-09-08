"""
LLM client and prompts for runtime, grounded explanations.

Uses DeepSeek's OpenAI-compatible API. The API key is read from the
environment (DEEPSEEK_API_KEY) — never hardcode it. In Kaggle, set it
via Add-ons > Secrets and export it into the environment before
importing this module; in the deployed app, set it as an environment
variable / Streamlit secret.

The LLM is given ONLY the structured evidence dict from
src/llm/evidence.py. It is explicitly instructed not to invent numbers
or override the risk tier / confidence already assigned by the
analytical layer.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env in the project root (local dev); no-op if it doesn't exist (e.g. Kaggle/deployed, where the key is set another way)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are a supply-chain analyst assistant for Flour Mills of Nigeria.
You explain inventory risk flags to operations staff in plain English.

Rules you MUST follow:
- Use ONLY the numbers given to you in the evidence. Never invent, estimate, or
  round numbers you were not given.
- Never change or second-guess the risk_tier or confidence value provided —
  those were already decided by the analytical system. Your job is only to
  explain WHY, using the evidence.
- Be concise: 2-4 sentences.
- If confidence is Low, say so plainly and explain why (e.g. new SKU / data quality),
  so the reader knows to treat the number with appropriate caution.
- Do not make causal claims the evidence doesn't support.
"""


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set in environment. "
            "In Kaggle: load via kaggle_secrets.UserSecretsClient and os.environ[...] = key."
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def explain_sku(evidence: dict, client: OpenAI | None = None) -> str:
    """Generate a runtime, grounded explanation for one SKU's risk flag."""
    client = client or get_client()

    user_prompt = f"Explain this SKU's inventory risk flag using only this evidence:\n\n{evidence}"

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
