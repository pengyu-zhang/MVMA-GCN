#!/usr/bin/env bash
# End-to-end pipeline check (a few minutes): prepares data if missing, then
# runs a short training with every component enabled on each dataset and
# re-evaluates the saved checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

if [ ! -d data/processed/acm ]; then
    bash scripts/prepare_data.sh
fi

for d in acm dblp imdb; do
    "$PYTHON" -m src.train --config configs/smoke.yaml --dataset "$d" --labelrate 20 \
        --output "outputs/smoke-${d}"
    "$PYTHON" -m src.evaluate --run "outputs/smoke-${d}"
done
echo "smoke test OK"
