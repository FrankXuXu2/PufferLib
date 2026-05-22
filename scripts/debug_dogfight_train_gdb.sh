#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/puffer_cuda_env.sh

TIMESTEPS="${DOGFIGHT_DEBUG_TIMESTEPS:-262144}"
OUT_DIR="${DOGFIGHT_DEBUG_OUT_DIR:-/tmp/dogfight_debug}"
mkdir -p "$OUT_DIR"

if ! command -v gdb >/dev/null 2>&1; then
    cat >&2 <<'INSTALL'
ERROR: gdb is not installed.

Install guidance:
  Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y gdb
  Fedora:        sudo dnf install -y gdb
  Arch:          sudo pacman -S gdb

This is an escalation path for Python/native-extension failures. Prefer
scripts/debug_dogfight_episode.sh for deterministic C env debugging first.
INSTALL
    exit 2
fi

./build.sh dogfight --debug

REPORT="$OUT_DIR/train_gdb_report.txt"
GDB_CMDS="$OUT_DIR/debug_dogfight_train.gdb"

{
    printf 'set pagination off\n'
    printf 'set confirm off\n'
    printf 'set print pretty on\n'
    printf 'run -m pufferlib.pufferl train dogfight --train.total-timesteps %s\n' "$TIMESTEPS"
    printf 'printf "\\n===== backtrace on stop/crash =====\\n"\n'
    printf 'bt full\n'
    printf 'info threads\n'
    printf 'thread apply all bt 12\n'
} > "$GDB_CMDS"

set +e
gdb -q -batch -x "$GDB_CMDS" --args .venv/bin/python >"$REPORT" 2>&1
STATUS=$?
set -e

echo "GDB exit status: $STATUS" >> "$REPORT"
echo "Wrote Python-extension GDB report to $REPORT"
exit "$STATUS"
