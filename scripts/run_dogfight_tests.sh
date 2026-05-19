#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/puffer_cuda_env.sh

if [[ ! -d ocean/dogfight ]]; then
    echo "ERROR: ocean/dogfight is not present yet. Port the dogfight4 env first." >&2
    exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv missing. Run: uv venv --python 3.12 .venv && uv sync" >&2
    exit 2
fi

./build.sh dogfight

if [[ -f ocean/dogfight/tests/run_all.sh ]]; then
    bash ocean/dogfight/tests/run_all.sh
else
    echo "ERROR: ocean/dogfight/tests/run_all.sh is missing." >&2
    exit 2
fi
