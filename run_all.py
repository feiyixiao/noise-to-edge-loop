"""Reproduce everything: noise manifest + all 30 runs + filter stats + quantization + table/figure.
Usage: python run_all.py            (resumable: finished runs in results/runs.csv are skipped)
Env: MLFLOW_DISABLE_AGENT_HINT=1 silences an MLflow hint."""
import subprocess
import sys
import time

t0 = time.time()
for mod in ("src.run_main", "src.quant", "src.report"):
    print(f"### {mod}", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", "-m", mod], check=True)
open("results/run_all_seconds.txt", "w").write(f"{time.time() - t0:.0f}\n")
