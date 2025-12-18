#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

LOG_DIR=logs
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
NOHUP_LOG="$LOG_DIR/$TS.log"

nohup python main.py \
    --mode train \
    --config configs/config.yaml \
  > "$NOHUP_LOG" 2>&1 &

PY_PID=$!

echo "Started training"
echo "  PID       : $PY_PID"
echo "  nohup log : $NOHUP_LOG"

sleep 5
tail -f "$NOHUP_LOG"
