#!/usr/bin/env bash
# Download the dataset tarball (GitHub Release asset) and build the processed
# layout under data/processed/. Pass --tarball <path> to use a local tar.gz.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}

"$PYTHON" -m src.download_data "$@"
"$PYTHON" -m src.prepare
