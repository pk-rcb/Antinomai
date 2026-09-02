"""
antinomai_results.py
====================
Runs the exact same Python/Pandas sandbox logic as Antinomai's portfolio_node
(extracted from app.py) against all 30 test portfolios.

This does NOT invoke the LLM — it only runs the verified math layer,
which is what we want to compare against the LLM-only baseline.

Outputs: benchmark/antinomai_results.json
"""

import os
import re
import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "test_portfolios.json"
OUT_FILE  = Path(__file__).parent / "antinomai_results.json"
STATIC_FX = 84.0

# ── Exact clones of app.py sandbox functions ─────────────────────────────────
def _get_live_fx_rate() -> tuple:
    """Try yfinance for live FX; fallback to static 84.0."""
    try:
        fx_hist = yf.Ticker("USDINR=X").history(period="1d")
        if not fx_hist.empty:
            rate = round(float(fx_hist["Close"].iloc[-1]), 4)
            print(f"[FX] Live rate: {rate}")
            return rate, True
    except Exception as e:
        print(f"[FX] Error: {e}")
    print(f"[FX] Using static fallback: {STATIC_FX}")
    return STATIC_FX, False

def _compute_beta(ticker: str, benchmark: str) -> Optional[float]:
    """Exact replica of app.py _compute_beta."""
    try:
        a = yf.Ticker(ticker).history(period="1y")["Close"].pct_change().dropna()
        b = yf.Ticker(benchmark).history(period="1y")["Close"].pct_change().dropna()
        df = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(df) < 20:
            return None
        cov = np.cov(df["a"], df["b"])
        return round(float(cov[0, 1] / cov[1, 1]), 4)
    except Exception:
        return None

def _compute_vol(ticker: str) -> Optional[float]:
    """Exact replica of app.py _compute_vol."""
    try:
        h = yf.Ticker(ticker).history(period="6mo")["Close"].pct_change().dropna()
        if len(h) < 20:
            return None
        return round(float(h.tail(30).std() * np.sqrt(252) * 100), 4)
    except Exception:
        return None

# ── Main ─────────────────────────────────────────────────────────────────────
def run_sandbox(holdings: list, fx_rate: float) -> dict:
    """
    Run the Python/Pandas math sandbox for one portfolio.
    Returns the same structure as ground_truth.py for easy comparison.
    """
    rows      = []
    total_usd = 0.0

    for h in holdings:
        ticker = h["ticker"]
        shares = h["shares"]
        is_inr = ticker.endswith(".NS")
        bench  = "^NSEI" if is_inr else "^GSPC"

        try:
            hist = yf.Ticker(ticker).history(period="1d")
            if hist.empty:
                print(f"  [SKIP] No data for {ticker}")
                continue
            price     = round(float(hist["Close"].iloc[-1]), 4)
            currency  = "INR" if is_inr else "USD"
            price_usd = round(price / fx_rate, 4) if is_inr else price
            value_usd = round(price_usd * shares, 4)
            beta      = _compute_beta(ticker, bench)
            vol       = _compute_vol(ticker)
            total_usd += value_usd

            rows.append({
                "ticker":    ticker,
                "shares":    shares,
                "price":     price,
                "currency":  currency,
                "price_usd": price_usd,
                "value_usd": value_usd,
                "beta":      beta,
                "vol_ann":   vol,
                "benchmark": bench,
            })
            print(f"  {ticker}: ${value_usd:,.2f} | Beta={beta} | Vol={vol}")
            time.sleep(0.2)

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    # Compute weights
    for r in rows:
        r["weight_pct"] = round((r["value_usd"] / total_usd) * 100, 4) if total_usd > 0 else 0.0

    # Weighted beta
    beta_rows = [r for r in rows if r["beta"] is not None]
    wtd_beta  = None
    if beta_rows:
        wtd_beta = round(
            sum(r["beta"] * r["weight_pct"] / 100 for r in beta_rows) /
            sum(r["weight_pct"] / 100 for r in beta_rows), 4
        )

    return {
        "total_usd": round(total_usd, 4),
        "wtd_beta":  wtd_beta,
        "assets":    rows,
    }


def main():
    portfolios = json.loads(DATA_FILE.read_text())
    fx_rate, fx_live = _get_live_fx_rate()
    results    = []

    print(f"\nRunning Antinomai sandbox on {len(portfolios)} portfolios...\n")

    for port in portfolios:
        pid = port["id"]
        print(f"[{pid}] {port['type']} portfolio...")

        sandbox_out = run_sandbox(port["holdings"], fx_rate)

        results.append({
            "id":            pid,
            "type":          port["type"],
            "query":         port["query"],
            "fx_rate":       fx_rate,
            "sandbox_result": sandbox_out,
        })

        total = sandbox_out["total_usd"]
        beta  = sandbox_out["wtd_beta"]
        print(f"  ✓ Total=${total:,.2f} | Wtd_Beta={beta}\n")

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Antinomai results saved → {OUT_FILE}")
    print(f"   Portfolios: {len(results)}")

if __name__ == "__main__":
    main()
