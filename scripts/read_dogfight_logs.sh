#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${DOGFIGHT_LOG_DIR:-/tmp/dogfight_logs}"
LOG="${1:-}"

if [[ -z "$LOG" ]]; then
    LOG="$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1 || true)"
fi

if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "ERROR: no log found. Pass a log path or set DOGFIGHT_LOG_DIR." >&2
    exit 2
fi

echo "== Log: $LOG =="
echo
echo "== Tail =="
tail -80 "$LOG"

echo
echo "== Errors =="
strings "$LOG" | grep -iE 'error|exception|traceback|assert|shape|crash|nan' | tail -80 || true

echo
echo "== Training markers =="
strings "$LOG" | grep -E 'CURRICULUM|curriculum_target|base_stage|stage|perf|SPS|player_ground|opponent_ground|avg_signed_bias|action_sat_[[:alnum:]_]+|score|clean|SELFPLAY|RATCHET|CHECKPOINT|EVAL|TRAIN|ERROR' | tail -120 || true
