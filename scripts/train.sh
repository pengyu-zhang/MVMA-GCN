#!/usr/bin/env bash
# Train MVMA-GCN.
#   scripts/train.sh                  # all datasets x label rates {20,40,60}
#   scripts/train.sh acm 20           # one dataset / label rate
# CONFIG selects the configuration (default: configs/default.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
CONFIG=${CONFIG:-configs/default.yaml}
NAME=$(basename "$CONFIG" .yaml)

if [ $# -ge 2 ]; then
    "$PYTHON" -m src.train --config "$CONFIG" --dataset "$1" --labelrate "$2" \
        --output "outputs/${NAME}-$1-l$2"
else
    for d in acm dblp imdb; do
        for l in 20 40 60; do
            "$PYTHON" -m src.train --config "$CONFIG" --dataset "$d" --labelrate "$l" \
                --output "outputs/${NAME}-${d}-l${l}" --quiet
        done
    done
fi
