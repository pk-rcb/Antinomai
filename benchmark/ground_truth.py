"""
ground_truth.py
===============
Pulls live yfinance data for all 30 test portfolios and computes
mathematically verified ground-truth values for:
  - Portfolio weight (%) per asset
  - Beta vs benchmark (^GSPC for USD, ^NSEI for INR)
  - Annualised volatility (30-day rolling std × √252 × 100)
  - FX conversion (INR price → USD)

Outputs: benchmark/ground_truth.json
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────────
STATIC_FX_RATE = 84.0          # Fallback USD/INR if live fetch fails
DATA_FILE      = Path(__file__).parent / "test_portfolios.json"
OUT_FILE       = Path(__file__).parent / "ground_truth.json"

# ── Helper functions (mirrors app.py exactly) ────────────────────────────────
def compute_beta(ticker: str, benchmark: str) -> float | None:
    try:
        a = yf.Ticker(ticker).history(period="1y")["Close"].pct_change().dropna()
        b = yf.Ticker(benchmark).history(period="1y")["Close"].pct_change().dropna()
        df = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(df) < 20:
            return None
        cov = np.cov(df["a"], df["b"])
        return round(float(cov[0, 1] / cov[1, 1]), 4)
    except Exception as e:
        print(f"  [WARN] Beta failed for {ticker}: {e}")
        return None

def compute_vol(ticker: str) -> float | None:
    try:
        h = yf.Ticker(ticker).history(period="6mo")["Close"].pct_change().dropna()
        if len(h) < 20:
            return None
        return round(float(h.tail(30).std() * np.sqrt(252) * 100), 4)
    except Exception as e:
        print(f"  [WARN] Vol failed for {ticker}: {e}")
        return None

def get_price(ticker: str) -> float | None:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            return None
        return round(float(hist["Close"].iloc[-1]), 4)
    except Exception as e:
        print(f"  [WARN] Price failed for {ticker}: {e}")
        return None

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    fx_rate = STATIC_FX_RATE

    # Try live FX from yfinance (USDINR=X)
    try:
        fx_hist = yf.Ticker("USDINR=X").history(period="1d")
        if not fx_hist.empty:
            fx_rate = round(float(fx_hist["Close"].iloc[-1]), 4)
            print(f"[FX] Live USD/INR rate: {fx_rate}")
        else:
            print(f"[FX] Fallback to static: {fx_rate}")
    except Exception as e:
        print(f"[FX] Error: {e}. Using static: {fx_rate}")

    portfolios = json.loads(DATA_FILE.read_text())
    results    = []

    for port in portfolios:
        pid    = port["id"]
        ptype  = port["type"]
        print(f"\n[{pid}] Processing {ptype} portfolio...")

        assets     = []
        total_usd  = 0.0

        for h in port["holdings"]:
            ticker = h["ticker"]
            shares = h["shares"]
            is_inr = ticker.endswith(".NS")
            bench  = "^NSEI" if is_inr else "^GSPC"

            print(f"  -> {ticker} ({shares} shares)...")
            price = get_price(ticker)
            if price is None:
                print(f"  [SKIP] No price for {ticker}")
                continue

            price_usd = round(price / fx_rate, 4) if is_inr else price
            value_usd = round(price_usd * shares, 4)
            beta      = compute_beta(ticker, bench)
            vol       = compute_vol(ticker)
            total_usd += value_usd

            assets.append({
                "ticker":    ticker,
                "shares":    shares,
                "price":     price,
                "currency":  "INR" if is_inr else "USD",
                "price_usd": price_usd,
                "value_usd": value_usd,
                "beta":      beta,
                "vol_ann":   vol,
                "benchmark": bench,
            })
            time.sleep(0.2)   # rate-limit yfinance

        # Compute weights
        for a in assets:
            a["weight_pct"] = round((a["value_usd"] / total_usd) * 100, 4) if total_usd > 0 else 0.0

        # Weighted beta
        beta_assets = [a for a in assets if a["beta"] is not None]
        wtd_beta    = None
        if beta_assets:
            wtd_beta = round(
                sum(a["beta"] * a["weight_pct"] / 100 for a in beta_assets) /
                sum(a["weight_pct"] / 100 for a in beta_assets), 4
            )

        results.append({
            "id":        pid,
            "type":      ptype,
            "query":     port["query"],
            "fx_rate":   fx_rate,
            "total_usd": round(total_usd, 4),
            "wtd_beta":  wtd_beta,
            "assets":    assets,
        })

        print(f"  ✓ Total: ${total_usd:,.2f} | Wtd Beta: {wtd_beta}")

    OUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Ground truth saved → {OUT_FILE}")
    print(f"   Portfolios: {len(results)} | Assets with data: {sum(len(r['assets']) for r in results)}")

if __name__ == "__main__":
    main()
