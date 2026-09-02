"""
run_benchmark.py
================
Orchestrator: runs all 4 benchmark steps in sequence.
Usage: python benchmark/run_benchmark.py
Requires: GROQ_API_KEY environment variable set
"""
import subprocess
import sys
from pathlib import Path

steps = [
    ("Step 1/4 — Building Ground Truth",      "benchmark/ground_truth.py"),
    ("Step 2/4 — Running LLM-Only Baseline",  "benchmark/baseline_llm.py"),
    ("Step 3/4 — Running Antinomai Sandbox",  "benchmark/antinomai_results.py"),
    ("Step 4/4 — Scoring & Generating Report","benchmark/score.py"),
]

root = Path(__file__).parent.parent

for title, script in steps:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(root / script)],
        cwd=str(root),
    )
    if result.returncode != 0:
        print(f"\n❌ {script} failed with exit code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

print("\n✅ Benchmark complete! Check benchmark/report.md for results.")
