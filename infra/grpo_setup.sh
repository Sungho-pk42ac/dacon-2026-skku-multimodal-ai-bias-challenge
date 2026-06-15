#!/usr/bin/env bash
# GRPO 전용 venv (llm_blender 충돌 회피 위해 LLaMA-Factory env와 분리).
# GPU 불필요(설치만). torch는 lf311과 동일 2.6/cu124. trl+transformers 버전 매칭 탐색.
set -o pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /workspace
echo "[grpo] venv 생성"
uv venv --python 3.11 /workspace/grpo_env
source /workspace/grpo_env/bin/activate
echo "[grpo] torch 2.6 (cu124)"
uv pip install --quiet torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
echo "[grpo] trl/transformers/deps"
uv pip install --quiet "transformers==4.52.4" "trl==0.19.0" accelerate peft "datasets>=3" qwen-vl-utils pillow pandas
echo "[grpo] import 테스트"
python - <<PYEOF
try:
    from trl import GRPOTrainer, GRPOConfig
    import trl, transformers, torch
    print("GRPO_IMPORT_OK trl", trl.__version__, "tf", transformers.__version__, "torch", torch.__version__)
except Exception as e:
    print("GRPO_IMPORT_FAIL", type(e).__name__, str(e)[:160])
PYEOF
echo "[grpo] SETUP_DONE"
