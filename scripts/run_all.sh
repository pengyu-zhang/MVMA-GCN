#!/usr/bin/env bash
# Full pipeline: data preparation, then training + evaluation of the
# recommended configuration on all datasets and label rates.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/prepare_data.sh
bash scripts/train.sh

for d in acm dblp imdb; do
    for l in 20 40 60; do
        bash scripts/evaluate.sh "outputs/default-${d}-l${l}"
    done
done
