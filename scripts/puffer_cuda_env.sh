#!/usr/bin/env bash

_puffer_prepend_path() {
    case ":${!1:-}:" in
        *":$2:"*) ;;
        *) export "$1=$2${!1:+:${!1}}" ;;
    esac
}

_PUFFER_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"

_puffer_prepend_path PATH "$_PUFFER_ENV_ROOT/.venv/bin"
_puffer_prepend_path PATH "$CUDA_HOME/bin"
_puffer_prepend_path PATH /usr/lib/wsl/lib

_puffer_prepend_path LD_LIBRARY_PATH /usr/lib/wsl/lib
_puffer_prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib64"

export CCACHE_DIR=/tmp/ccache
export CC="${CC:-clang}"

unset _PUFFER_ENV_ROOT
unset -f _puffer_prepend_path
