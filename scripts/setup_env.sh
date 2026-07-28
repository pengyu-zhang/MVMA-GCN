#!/usr/bin/env bash
# Install dependencies into the current Python environment.
# CUDA_TAG selects the PyTorch build: cu132 (default), cu126, ... or "cpu".
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
CUDA_TAG=${CUDA_TAG:-cu132}

echo "installing torch (${CUDA_TAG}) ..."
"$PYTHON" -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
echo "installing remaining requirements ..."
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
