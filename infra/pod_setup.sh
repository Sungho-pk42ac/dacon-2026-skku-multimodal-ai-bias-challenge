#!/usr/bin/env bash
set -euo pipefail
cd /workspace
echo "[setup] torch before: $(python -c 'import torch;print(torch.__version__)')"
[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -q -e ".[metrics]"
pip install -q qwen-vl-utils wandb
echo "[setup] torch after: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"
echo "[setup] llamafactory: $(llamafactory-cli version 2>&1 | tail -1)"
echo "[setup] DONE"
