"""
score.py
========
Compares Antinomai and LLM-baseline results against ground truth.
Computes per-metric hallucination rates and the overall reduction %.

Definition of hallucination:
  |predicted - ground_truth| / |ground_truth| > THRESHOLD (default 5%)

Outputs:
  - benchmark/report.md    (human-readable summary)
  - benchmark/scores.csv   (raw per-asset scores)
"""

import json
import csv
import sys
import math
from pathlib import Path
from datetime import datetime

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ──────────────────────────────────────────────────────────────────
THRESHOLD       = 0.05        # 5% relative error = hallucination
GT_FILE         = Path(__file__).parent / "ground_truth.json"
BASELINE_FILE   = Path(__file__).parent / "baseline_results.json"
ANTINOMAI_FILE  = Path(__file__).parent / "antinomai_results.json"
REPORT_FILE     = Path(__file__).parent / "report.md"
SCORES_FILE     = Path(__file__).parent / "scores.csv"

METRICS = ["weight_pct", "beta", "vol_ann", "price_usd"]
METRIC_LABELS = {
    "weight_pct": "Portfolio Weight (%)",
    "beta":       "Beta (vs Benchmark)",
    "vol_ann":    "Annualised Volatility (%)",
    "price_usd":  "Price in USD (FX Conversion)",
}

# ── Scoring helpers ───────────────────────────────────────────────────────────
def relative_error(predicted, ground_truth) -> float | None:
    if predicted is None or ground_truth is None:
        return None
    if ground_truth == 0:
        return None
    return abs(predicted - ground_truth) / abs(ground_truth)

def is_hallucination(error: float | None) -> bool:
    if error is None:
        return True    # Missing value = hallucination
    return error > THRESHOLD

def find_gt_asset(gt_portfolio: dict, ticker: str) -> dict | None:
    for a in gt_portfolio["assets"]:
        if a["ticker"] == ticker:
            return a
    return None

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    gt_data    = json.loads(GT_FILE.read_text())
    base_data  = json.loads(BASELINE_FILE.read_text())
    anti_data  = json.loads(ANTINOMAI_FILE.read_text())

    # Index by portfolio ID
    gt_by_id   = {p["id"]: p for p in gt_data}
    base_by_id = {p["id"]: p for p in base_data}
    anti_by_id = {p["id"]: p for p in anti_data}

    csv_rows   = []   # For scores.csv
    metric_scores = {m: {"baseline": [], "antinomai": []} for m in METRICS}
    port_scores   = []

    for pid, gt in gt_by_id.items():
        base = base_by_id.get(pid)
        anti = anti_by_id.get(pid)

        if not base or not anti:
            print(f"[SKIP] Missing data for {pid}")
            continue

        base_assets  = (base.get("llm_response") or {}).get("assets", [])
        anti_assets  = (anti.get("sandbox_result") or {}).get("assets", [])

        # Index by ticker
        base_by_ticker = {a["ticker"]: a for a in base_assets}
        anti_by_ticker = {a["ticker"]: a for a in anti_assets}

        port_base_halluc = {m: [] for m in METRICS}
        port_anti_halluc = {m: [] for m in METRICS}

        for gt_asset in gt["assets"]:
            ticker = gt_asset["ticker"]
            ba     = base_by_ticker.get(ticker)
            aa     = anti_by_ticker.get(ticker)

            for metric in METRICS:
                gt_val   = gt_asset.get(metric)
                base_val = ba.get(metric) if ba else None
                anti_val = aa.get(metric) if aa else None

                base_err  = relative_error(base_val, gt_val)
                anti_err  = relative_error(anti_val, gt_val)
                base_hall = is_hallucination(base_err)
                anti_hall = is_hallucination(anti_err)

                metric_scores[metric]["baseline"].append(int(base_hall))
                metric_scores[metric]["antinomai"].append(int(anti_hall))
                port_base_halluc[metric].append(int(base_hall))
                port_anti_halluc[metric].append(int(anti_hall))

                csv_rows.append({
                    "portfolio_id":     pid,
                    "portfolio_type":   gt["type"],
                    "ticker":           ticker,
                    "metric":           metric,
                    "gt_value":         gt_val,
                    "baseline_value":   base_val,
                    "antinomai_value":  anti_val,
                    "baseline_error":   round(base_err * 100, 2) if base_err is not None else "N/A",
                    "antinomai_error":  round(anti_err * 100, 2) if anti_err is not None else "N/A",
                    "baseline_halluc":  base_hall,
                    "antinomai_halluc": anti_hall,
                })

        # Per-portfolio summary
        all_base  = [v for m in METRICS for v in port_base_halluc[m]]
        all_anti  = [v for m in METRICS for v in port_anti_halluc[m]]
        port_scores.append({
            "id":            pid,
            "type":          gt["type"],
            "base_rate":     round(sum(all_base) / len(all_base) * 100, 1) if all_base else 0,
            "anti_rate":     round(sum(all_anti) / len(all_anti) * 100, 1) if all_anti else 0,
        })

    # ── Per-metric summary ────────────────────────────────────────────────────
    summary = {}
    for metric in METRICS:
        b = metric_scores[metric]["baseline"]
        a = metric_scores[metric]["antinomai"]
        base_rate = round(sum(b) / len(b) * 100, 1) if b else 0.0
        anti_rate = round(sum(a) / len(a) * 100, 1) if a else 0.0
        reduction = round((base_rate - anti_rate) / base_rate * 100, 1) if base_rate > 0 else 0.0
        summary[metric] = {
            "label":      METRIC_LABELS[metric],
            "base_rate":  base_rate,
            "anti_rate":  anti_rate,
            "reduction":  reduction,
            "n_checks":   len(b),
        }

    all_b   = [v for m in METRICS for v in metric_scores[m]["baseline"]]
    all_a   = [v for m in METRICS for v in metric_scores[m]["antinomai"]]
    overall_base = round(sum(all_b) / len(all_b) * 100, 1) if all_b else 0
    overall_anti = round(sum(all_a) / len(all_a) * 100, 1) if all_a else 0
    overall_red  = round((overall_base - overall_anti) / overall_base * 100, 1) if overall_base > 0 else 0

    # ── Write scores.csv ─────────────────────────────────────────────────────
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(SCORES_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[OK] Scores CSV -> {SCORES_FILE}")

    # ── Write report.md ───────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# Arithmetic Hallucination Benchmark Report
**Antinomai Portfolio Risk Engine vs LLM-Only Baseline**
_Generated: {now}_

---

## Methodology

- **Test Set**: 30 portfolios (10 USD-only, 10 INR-only, 10 mixed)
- **Ground Truth**: yfinance live prices + NumPy/Pandas computation
- **Baseline**: Llama-3.3-70b-versatile, no tools, computes math itself
- **Antinomai**: Python/Pandas sandbox → verified numbers → LLM narrative only
- **Hallucination Threshold**: Relative error > 5% = hallucination

---

## Results Summary

| Metric | LLM Baseline Error Rate | Antinomai Error Rate | Reduction |
|---|---|---|---|
"""
    for metric in METRICS:
        s = summary[metric]
        report += f"| {s['label']} | {s['base_rate']}% | {s['anti_rate']}% | **{s['reduction']}%** |\n"

    report += f"| **Overall** | **{overall_base}%** | **{overall_anti}%** | **{overall_red}%** |\n"

    report += f"""
---

## Interpretation

Antinomai's Portfolio Node achieves a **{overall_red}% reduction in arithmetic hallucination errors**
compared to a naive LLM-only approach, by separating quantitative computation from language generation:

- All Beta, Volatility, Weight, and FX calculations are performed in a **pure Python/Pandas/NumPy sandbox**.
- The LLM receives only the **verified, immutable math payload** as a string.
- The LLM's role is **narrative reasoning only** — a task it reliably performs.

---

## Per-Portfolio Breakdown

| Portfolio | Type | Baseline Error Rate | Antinomai Error Rate |
|---|---|---|---|
"""
    for ps in sorted(port_scores, key=lambda x: x["base_rate"], reverse=True):
        report += f"| {ps['id']} | {ps['type']} | {ps['base_rate']}% | {ps['anti_rate']}% |\n"

    report += f"""
---

## Files

- `ground_truth.json` — Verified yfinance-computed values
- `baseline_results.json` — Raw LLM-only outputs
- `antinomai_results.json` — Antinomai sandbox outputs
- `scores.csv` — Per-asset, per-metric error details

_Threshold: {THRESHOLD*100:.0f}% relative error = hallucination_
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"[OK] Report saved -> {REPORT_FILE}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ARITHMETIC HALLUCINATION BENCHMARK — RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Antinomai':>10} {'Reduction':>10}")
    print("-" * 60)
    for metric in METRICS:
        s = summary[metric]
        print(f"{s['label']:<30} {s['base_rate']:>9}% {s['anti_rate']:>9}% {s['reduction']:>9}%")
    print("-" * 60)
    print(f"{'OVERALL':<30} {overall_base:>9}% {overall_anti:>9}% {overall_red:>9}%")
    print("=" * 60)
    print(f"\n[TARGET] Antinomai reduces arithmetic hallucinations by {overall_red}%")


if __name__ == "__main__":
    main()
