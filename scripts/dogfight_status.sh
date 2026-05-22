#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Branch =="
git branch --show-current

echo
echo "== Working tree =="
git status --short

echo
echo "== Remotes =="
git remote -v

echo
echo "== Dogfight files in working repo =="
if [[ -d ocean/dogfight ]]; then
    find ocean/dogfight -maxdepth 3 -type f | sort
else
    echo "ocean/dogfight is not present yet"
fi

echo
echo "== Reference dogfight4 =="
find /home/claude/dogfight4/ocean/dogfight -maxdepth 2 -type f | sort | sed -n '1,120p'

echo
echo "== Reference dogfight3 =="
find /home/claude/dogfight3/pufferlib/ocean/dogfight -maxdepth 2 -type f | sort | sed -n '1,160p'
