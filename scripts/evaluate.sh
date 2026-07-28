#!/usr/bin/env bash
# Evaluate a trained run from its checkpoint.
#   scripts/evaluate.sh outputs/default-acm-l20
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

if [ $# -lt 1 ]; then
    echo "usage: scripts/evaluate.sh <run_dir>" >&2
    exit 1
fi
"$PYTHON" -m src.evaluate --run "$1"
