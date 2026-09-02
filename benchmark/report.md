# Arithmetic Hallucination Benchmark Report
**Antinomai Portfolio Risk Engine vs LLM-Only Baseline**
_Generated: 2026-07-04 00:32_

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
| Portfolio Weight (%) | 96.6% | 0.0% | **100.0%** |
| Beta (vs Benchmark) | 94.3% | 0.0% | **100.0%** |
| Annualised Volatility (%) | 85.1% | 0.0% | **100.0%** |
| Price in USD (FX Conversion) | 100.0% | 0.0% | **100.0%** |
| **Overall** | **94.0%** | **0.0%** | **100.0%** |

---

## Interpretation

Antinomai's Portfolio Node achieves a **100.0% reduction in arithmetic hallucination errors**
compared to a naive LLM-only approach, by separating quantitative computation from language generation:

- All Beta, Volatility, Weight, and FX calculations are performed in a **pure Python/Pandas/NumPy sandbox**.
- The LLM receives only the **verified, immutable math payload** as a string.
- The LLM's role is **narrative reasoning only** — a task it reliably performs.

---

## Per-Portfolio Breakdown

| Portfolio | Type | Baseline Error Rate | Antinomai Error Rate |
|---|---|---|---|
| USD_03 | USD | 100.0% | 0.0% |
| USD_05 | USD | 100.0% | 0.0% |
| USD_07 | USD | 100.0% | 0.0% |
| USD_08 | USD | 100.0% | 0.0% |
| USD_09 | USD | 100.0% | 0.0% |
| INR_01 | INR | 100.0% | 0.0% |
| INR_03 | INR | 100.0% | 0.0% |
| INR_04 | INR | 100.0% | 0.0% |
| INR_06 | INR | 100.0% | 0.0% |
| INR_07 | INR | 100.0% | 0.0% |
| INR_10 | INR | 100.0% | 0.0% |
| MIX_01 | MIXED | 100.0% | 0.0% |
| MIX_02 | MIXED | 100.0% | 0.0% |
| MIX_04 | MIXED | 100.0% | 0.0% |
| MIX_06 | MIXED | 100.0% | 0.0% |
| MIX_07 | MIXED | 100.0% | 0.0% |
| USD_02 | USD | 91.7% | 0.0% |
| USD_04 | USD | 91.7% | 0.0% |
| USD_06 | USD | 91.7% | 0.0% |
| USD_10 | USD | 91.7% | 0.0% |
| INR_05 | INR | 91.7% | 0.0% |
| INR_09 | INR | 91.7% | 0.0% |
| MIX_03 | MIXED | 91.7% | 0.0% |
| MIX_05 | MIXED | 91.7% | 0.0% |
| MIX_09 | MIXED | 91.7% | 0.0% |
| USD_01 | USD | 83.3% | 0.0% |
| INR_02 | INR | 83.3% | 0.0% |
| INR_08 | INR | 83.3% | 0.0% |
| MIX_08 | MIXED | 75.0% | 0.0% |
| MIX_10 | MIXED | 75.0% | 0.0% |

---

## Files

- `ground_truth.json` — Verified yfinance-computed values
- `baseline_results.json` — Raw LLM-only outputs
- `antinomai_results.json` — Antinomai sandbox outputs
- `scores.csv` — Per-asset, per-metric error details

_Threshold: 5% relative error = hallucination_
