#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ACTIVATE=".venv/bin/activate"
HELPER="/home/claude/PufferLib/scripts/puffer_cuda_env.sh"

if [[ ! -f "$ACTIVATE" ]]; then
    echo "ERROR: $ACTIVATE not found. Create the venv first." >&2
    exit 2
fi

if grep -Fq "$HELPER" "$ACTIVATE"; then
    echo "Puffer CUDA venv hook already installed in $ACTIVATE"
    exit 0
fi

tmp="$(mktemp)"
awk -v helper="$HELPER" '
    {
        print
        if ($0 == "export PATH" && ! inserted) {
            print ""
            print "if [ -f \"" helper "\" ] ; then"
            print "    source \"" helper "\""
            print "fi"
            inserted = 1
        }
    }
    END {
        if (! inserted) {
            exit 1
        }
    }
' "$ACTIVATE" > "$tmp" || {
    rm -f "$tmp"
    echo "ERROR: could not find activation PATH export insertion point." >&2
    exit 2
}

cat "$tmp" > "$ACTIVATE"
rm -f "$tmp"
echo "Installed Puffer CUDA venv hook in $ACTIVATE"
