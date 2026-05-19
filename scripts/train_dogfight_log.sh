#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STEPS="${1:-300000}"
if [[ $# -gt 0 ]]; then
    shift
fi
LOG_DIR="${DOGFIGHT_LOG_DIR:-/tmp/dogfight_logs}"
mkdir -p "$LOG_DIR"

source scripts/puffer_cuda_env.sh

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv missing. Run: uv venv --python 3.12 .venv && uv sync" >&2
    exit 2
fi

if [[ ! -f config/dogfight.ini ]]; then
    echo "ERROR: config/dogfight.ini is missing. Port config from dogfight4 first." >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/dogfight_train_${STEPS}_${STAMP}.log"
echo "Logging to $LOG"

./build.sh dogfight
cmd=(.venv/bin/python -m pufferlib.pufferl train dogfight --train.total-timesteps "$STEPS" "$@")
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "$LOG"
