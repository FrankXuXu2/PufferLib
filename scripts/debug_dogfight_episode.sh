#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/puffer_cuda_env.sh

MODE="run"
STEPS="300"
SEED="42"
STAGE="0"
OBS_SCHEME="0"
ACTION="neutral"
ACTION_VALUES=""
BREAK_TICK=""
WATCH_REWARD="0"
OUT_DIR="/tmp/dogfight_debug"

usage() {
    cat >&2 <<'USAGE'
Usage: scripts/debug_dogfight_episode.sh [options]

Options:
  --mode run|gdb
  --steps N
  --seed N
  --stage N
  --obs-scheme N
  --action neutral|constant
  --action-values throttle,elevator,aileron,rudder,trigger
  --break-tick N              In gdb mode, stop at pre-step N after writing context
  --watch-reward              In gdb mode, stop on writes to env->rewards[0]
  --out-dir DIR               Default: /tmp/dogfight_debug

Examples:
  scripts/debug_dogfight_episode.sh --mode run --steps 300 --seed 42 --stage 0 --action neutral
  scripts/debug_dogfight_episode.sh --mode gdb --steps 300 --seed 42 --stage 0 --break-tick 120
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?}"; shift 2 ;;
        --steps) STEPS="${2:?}"; shift 2 ;;
        --seed) SEED="${2:?}"; shift 2 ;;
        --stage) STAGE="${2:?}"; shift 2 ;;
        --obs-scheme) OBS_SCHEME="${2:?}"; shift 2 ;;
        --action) ACTION="${2:?}"; shift 2 ;;
        --action-values) ACTION_VALUES="${2:?}"; shift 2 ;;
        --break-tick) BREAK_TICK="${2:?}"; shift 2 ;;
        --watch-reward) WATCH_REWARD="1"; shift ;;
        --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "$MODE" != "run" && "$MODE" != "gdb" ]]; then
    echo "ERROR: --mode must be run or gdb" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"

SRC="ocean/dogfight/tests/dogfight_debug_episode.c"
BIN="$OUT_DIR/dogfight_debug_episode"
REPORT="$OUT_DIR/gdb_report.txt"
GDB_CMDS="$OUT_DIR/debug_dogfight_episode.gdb"
RAYLIB_INC="raylib-5.5_linux_amd64/include"
RAYLIB_LIB="raylib-5.5_linux_amd64/lib/libraylib.a"

if [[ ! -f "$RAYLIB_LIB" ]]; then
    echo "ERROR: raylib static lib not found at $RAYLIB_LIB" >&2
    echo "       Run: ./build.sh dogfight" >&2
    exit 2
fi

clang -g -O0 -fno-omit-frame-pointer \
    -Wall -Wextra -Wno-unused-function -Wno-unused-parameter \
    -Wno-unused-variable -Wno-sign-compare \
    -I ocean/dogfight -I "$RAYLIB_INC" \
    "$SRC" "$RAYLIB_LIB" -lm -lpthread -ldl \
    -o "$BIN"

HARNESS_ARGS=(
    --steps "$STEPS"
    --seed "$SEED"
    --stage "$STAGE"
    --obs-scheme "$OBS_SCHEME"
    --action "$ACTION"
    --out-dir "$OUT_DIR"
)
if [[ -n "$ACTION_VALUES" ]]; then
    HARNESS_ARGS+=(--action-values "$ACTION_VALUES")
fi

if [[ "$MODE" == "run" ]]; then
    "$BIN" "${HARNESS_ARGS[@]}"
    echo "Wrote deterministic trace under $OUT_DIR"
    exit 0
fi

if ! command -v gdb >/dev/null 2>&1; then
    cat >&2 <<'INSTALL'
ERROR: gdb is not installed.

Install guidance:
  Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y gdb
  Fedora:        sudo dnf install -y gdb
  Arch:          sudo pacman -S gdb

Dependency policy: this script only detects missing tools. Request approval
before installing system packages.
INSTALL
    exit 2
fi

cp scripts/debug_dogfight_episode.gdb "$GDB_CMDS"
if [[ -n "$BREAK_TICK" ]]; then
    printf '\nset $dogfight_break_tick = %s\n' "$BREAK_TICK" >> "$GDB_CMDS"
fi
if [[ "$WATCH_REWARD" == "1" ]]; then
    printf '\nset $dogfight_watch_reward = 1\n' >> "$GDB_CMDS"
fi

{
    printf '\nrun'
    printf '\n'
    printf 'printf "\\n===== final status =====\\n"\n'
    printf 'printf "inferior exited or stopped; see trace above\\n"\n'
    printf 'bt full\n'
    printf 'info registers\n'
} >> "$GDB_CMDS"

set +e
gdb -q -batch -x "$GDB_CMDS" --args "$BIN" "${HARNESS_ARGS[@]}" >"$REPORT" 2>&1
STATUS=$?
set -e

echo "GDB exit status: $STATUS" >> "$REPORT"
echo "Wrote GDB report to $REPORT"
exit "$STATUS"
