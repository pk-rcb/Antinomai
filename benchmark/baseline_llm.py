"""
baseline_llm.py
===============
Sends each portfolio query to a bare Llama-3.3-70b and asks it to compute
portfolio math ITSELF (no Python sandbox, no tools) — simulating what a naive
LLM-only approach would produce.

For each portfolio the LLM is asked to compute:
  1. Weight (%) of each asset
  2. Beta of each asset vs its benchmark
  3. Annualised volatility of each asset
  4. Price in USD (after FX conversion for INR assets)

The LLM receives only the query text + share counts. It must estimate
prices from its training knowledge and compute all numbers itself.

Outputs: benchmark/baseline_results.json
Requires: GROQ_API_KEY environment variable
"""

import os
import re
import sys
import json
import time
from pathlib import Path

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ── Config ──────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "test_portfolios.json"
OUT_FILE  = Path(__file__).parent / "baseline_results.json"

SYSTEM_PROMPT = """You are a quantitative portfolio analyst. The user will provide portfolio holdings.
You must compute ALL of the following values YOURSELF using your knowledge of stock prices:

For EACH asset:
1. Current stock price (estimate from your knowledge)
2. Total value in USD (price × shares; convert INR to USD using ~84 INR/USD if needed)
3. Portfolio weight % = (asset_value / total_portfolio_value) × 100
4. Beta vs benchmark (^GSPC for US stocks, ^NSEI for Indian .NS stocks) — use your knowledge
5. Annualised volatility % = estimate from your knowledge of the stock's historical behaviour

Then compute:
6. Total portfolio value in USD
7. Weighted beta = sum(beta_i × weight_i) for all assets

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
  "total_usd": <number>,
  "wtd_beta": <number>,
  "assets": [
    {
      "ticker": "<TICKER>",
      "shares": <number>,
      "price": <number>,
      "currency": "USD" or "INR",
      "price_usd": <number>,
      "value_usd": <number>,
      "weight_pct": <number>,
      "beta": <number>,
      "vol_ann": <number>
    }
  ]
}"""


def query_llm_baseline(llm: ChatGroq, query: str, holdings: list) -> dict | None:
    holdings_str = "\n".join(
        f"  - {h['ticker']}: {h['shares']} shares"
        for h in holdings
    )
    user_msg = f"Portfolio holdings:\n{holdings_str}\n\nOriginal query: \"{query}\""

    try:
        resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = resp.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse failed: {e}")
        print(f"  Raw response: {resp.content[:300]}")
        return None
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return None


def main():
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not set. Export it before running.")

    llm       = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=groq_key)
    portfolios = json.loads(DATA_FILE.read_text())
    results    = []
    failed     = 0

    print(f"Running LLM-only baseline on {len(portfolios)} portfolios...\n")

    for port in portfolios:
        pid   = port["id"]
        print(f"[{pid}] Querying LLM... ", end="", flush=True)

        parsed = query_llm_baseline(llm, port["query"], port["holdings"])

        if parsed:
            results.append({
                "id":            pid,
                "type":          port["type"],
                "query":         port["query"],
                "llm_response":  parsed,
                "parse_success": True,
            })
            total = parsed.get("total_usd", "?")
            beta  = parsed.get("wtd_beta", "?")
            print(f"[OK] Total=${total}  Wtd_Beta={beta}")
        else:
            results.append({
                "id":            pid,
                "type":          port["type"],
                "query":         port["query"],
                "llm_response":  None,
                "parse_success": False,
            })
            failed += 1
            print("[FAIL] (parse failed)")

        time.sleep(1.5)   # Respect Groq rate limits

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Baseline results saved → {OUT_FILE}")
    print(f"   Success: {len(portfolios) - failed}/{len(portfolios)} | Failed: {failed}")

if __name__ == "__main__":
    main()
