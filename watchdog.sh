#!/bin/bash
# Restarts src.run_main if it died before finishing (resumable via runs.csv + per-epoch checkpoints).
# When training is complete (results/train_done exists) it runs quantization and the report once, then exits.
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
while [ ! -f results/train_done ]; do
  if ! pgrep -f "src.run_main" >/dev/null; then
    echo "$(date -u +%FT%TZ) restart run_main" >> results/watchdog.log
    nohup python3 -W ignore -m src.run_main >> results/main_log.txt 2>&1 &
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) training complete; quant + report" >> results/watchdog.log
python3 -W ignore -m src.quant > results/quant_log.txt 2>&1
python3 -W ignore -m src.report > results/report_log.txt 2>&1
echo "$(date -u +%FT%TZ) done" >> results/watchdog.log
