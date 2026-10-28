#!/bin/bash
# Restarts src.run_5b if it died before finishing (resumable via runs_5b.csv + per-epoch checkpoints).
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
while [ ! -f results/train_done_5b ]; do
  if ! pgrep -f "ignore -m src.run_5b" >/dev/null; then
    echo "$(date -u +%FT%TZ) restart run_5b" >> results/watchdog_5b.log
    nohup python3 -W ignore -m src.run_5b >> results/main_log_5b.txt 2>&1 &
  fi
  sleep 60
done
echo "$(date -u +%FT%TZ) 5b complete" >> results/watchdog_5b.log
