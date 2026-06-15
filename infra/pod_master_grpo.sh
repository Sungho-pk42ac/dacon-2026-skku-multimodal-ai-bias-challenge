#!/usr/bin/env bash
# GRPO 오케스트레이터: v3 SFT 완료 대기 → v3 HF업로드 → GRPO(grpo_env, LoRA) → 병합 → 자체검증 → CSV(조건부) → HF.
set -o pipefail
cd /workspace/dacon-bias-challenge
LOG=/workspace/master_grpo.log
exec >> "$LOG" 2>&1
echo "=== MASTER GRPO START $(date) ==="

echo "[0] v3 SFT(eval) 완료 대기 (GPU 확보)..."
while ! test -f /workspace/V3_EVAL_DONE; do sleep 30; done
while pgrep -f "llamafactory|baseline_eval|make_submission" >/dev/null; do sleep 30; done
echo "[0] GPU 확보됨."

# v3 HF 업로드 (lf311 env)
echo "[1] v3 HF 업로드..."
( source /workspace/lf311/bin/activate; export PATH="$HOME/.local/bin:$PATH"
  hf repo create psh3333/dacon-skku-bias-vlm-v3 --repo-type model --private 2>/dev/null
  hf upload psh3333/dacon-skku-bias-vlm-v3 outputs/merged_v3 . --repo-type model 2>&1 | tail -1 ) || echo "[1] v3 HF WARN"

# GRPO 학습 (grpo_env)
echo "[2] GRPO 학습 시작 (grpo_env)..."
source /workspace/grpo_env/bin/activate
export PATH="$HOME/.local/bin:$PATH"; export WANDB_PROJECT=dacon-bias-challenge
rm -rf outputs/grpo_v4
python train/grpo/train_grpo_vlm.py --base outputs/merged_v2 --out outputs/grpo_v4 --n 1200 --steps 200 --num_gen 4 || { echo "[2] GRPO ABORT"; touch /workspace/MASTER_GRPO_DONE; exit 1; }
echo "[2] GRPO 학습 완료."

# 병합 (lf311 llamafactory): merged_v2 + grpo_v4 어댑터 → merged_grpo
echo "[3] GRPO 어댑터 병합..."
source /workspace/lf311/bin/activate; export PATH="$HOME/.local/bin:$PATH"
rm -rf outputs/merged_grpo
llamafactory-cli export --model_name_or_path outputs/merged_v2 --adapter_name_or_path outputs/grpo_v4 \
  --template qwen2_vl --finetuning_type lora --export_dir outputs/merged_grpo --export_size 5 --trust_remote_code true \
  || { echo "[3] merge ABORT"; touch /workspace/MASTER_GRPO_DONE; exit 1; }

echo "[4] 자체검증 평가..."
python inference/baseline_eval.py --model outputs/merged_grpo --val data/bbq_val.jsonl --limit 0 --out data/grpo_preds.csv

echo "[5] submission_grpo.csv 생성 (전체 8500)..."
python inference/make_submission.py --model outputs/merged_grpo --test data_build/test_full.csv \
  --images_dir /nonexistent --fallback_image data/placeholder.jpg --out submissions/submission_grpo.csv

echo "[6] GRPO 모델 HF 업로드..."
hf repo create psh3333/dacon-skku-bias-vlm-grpo --repo-type model --private 2>/dev/null
hf upload psh3333/dacon-skku-bias-vlm-grpo outputs/merged_grpo . --repo-type model 2>&1 | tail -1 || echo "[6] grpo HF WARN"

echo "RESULT_GRPO_BA: $(grep -aoE 'Balanced Accuracy: [0-9.]+' ${LOG} | tail -1)"
echo "=== MASTER GRPO DONE $(date) ==="
touch /workspace/MASTER_GRPO_DONE
