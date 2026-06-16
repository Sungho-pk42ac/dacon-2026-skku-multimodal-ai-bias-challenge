#!/usr/bin/env bash
set -euo pipefail
cd /workspace
echo "[lf] uv 설치"
if ! command -v uv >/dev/null 2>&1; then
  (curl -LsSf https://astral.sh/uv/install.sh | sh) || (wget -qO- https://astral.sh/uv/install.sh | sh)
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo "[lf] py3.11 venv 생성"
uv venv --python 3.11 /workspace/lf311
source /workspace/lf311/bin/activate
python --version
echo "[lf] torch 2.6 (cu124) 설치"
uv pip install --quiet torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
echo "[lf] LLaMA-Factory clone+install"
[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
uv pip install --quiet -e ".[metrics]"
uv pip install --quiet qwen-vl-utils wandb
echo "[lf] torch after: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"
echo "[lf] llamafactory: $(llamafactory-cli version 2>&1 | tail -1)"
echo "LF311_DONE"
