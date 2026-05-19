#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/puffer_cuda_env.sh

if [[ ! -x .venv/bin/python ]]; then
    echo "ERROR: .venv missing. Run: uv venv --python 3.12 .venv && uv sync" >&2
    exit 2
fi

echo "== GPU =="
nvidia-smi -L
nvcc --version | sed -n '1,4p'
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if not torch.cuda.is_available():
    raise SystemExit(3)
PY

echo "== Build breakout CUDA backend =="
./build.sh breakout

echo "== Train breakout smoke =="
.venv/bin/python -m pufferlib.pufferl train breakout --train.total-timesteps 300000
