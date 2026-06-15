#!/usr/bin/env bash
# v5 오케스트레이터 — GRPO on v4 (회의: 깨끗한 val에서 v4 넘는지 정직 측정).
# 흐름: v4완료 대기 → GRPO(grpo3_env, merged_v4 기점, 누출배제) → 병합(lf311) → 정직한 BA(bbq_val_clean) → 제출 → HF → v4 vs v5 비교.
set -o pipefail
cd /workspace/dacon-bias-challenge
LOG=/workspace/master_v5.log
exec >> "$LOG" 2>&1
echo "=== MASTER v5 (GRPO on Qwen3-VL) START $(date) ==="

echo "[0] v4 완료 + GPU 확보 대기..."
while ! test -f /workspace/MASTER_v4_DONE; do sleep 30; done
while pgrep -f "make_submission|baseline_eval|llamafactory|train_grpo" >/dev/null; do sleep 30; done
test -d outputs/merged_v4 || { echo "ABORT: merged_v4 없음(v4 실패?)"; exit 1; }
test -f data/val_ids.json || { echo "ABORT: val_ids.json 없음(누출배제 불가)"; exit 1; }
echo "[0] GPU 확보됨 $(date)."

echo "[1] GRPO 학습 (grpo3_env, merged_v4 기점, 누출배제)..."
source /workspace/grpo3_env/bin/activate
export PATH="$HOME/.local/bin:$PATH"; export WANDB_PROJECT=dacon-bias-challenge
rm -rf outputs/grpo_v5
python train/grpo/train_grpo_vlm.py --base outputs/merged_v4 --out outputs/grpo_v5 \
  --val_ids data/val_ids.json --n 1500 --steps 250 --num_gen 4 --max_completion_length 8 \
  || { echo "[1] GRPO ABORT"; touch /workspace/MASTER_v5_DONE; exit 1; }

echo "[2] 병합 (lf311, merged_v4 + grpo_v5 어댑터 → merged_v5)..."
source /workspace/lf311/bin/activate; export PATH="$HOME/.local/bin:$PATH"
rm -rf outputs/merged_v5
llamafactory-cli export --model_name_or_path outputs/merged_v4 --adapter_name_or_path outputs/grpo_v5 \
  --template qwen3_vl_nothink --finetuning_type lora --export_dir outputs/merged_v5 --export_size 5 --trust_remote_code true \
  || { echo "[2] merge ABORT"; touch /workspace/MASTER_v5_DONE; exit 1; }

echo "[3] 정직한 BA (bbq_val_clean, max_new_tokens=8) — v4와 동일 검증셋..."
python inference/baseline_eval.py --model outputs/merged_v5 --val data/bbq_val_clean.jsonl --limit 0 \
  --max_new_tokens 8 --out data/v5_preds.csv

echo "[4] submission_v5.csv (전체 8500, max_new_tokens=4)..."
python inference/make_submission.py --model outputs/merged_v5 --test data_build/test_full.csv \
  --images_dir /nonexistent --fallback_image data/placeholder.jpg \
  --out submissions/submission_v5.csv --max_new_tokens 4

echo "[5] v5 HF 업로드..."
hf repo create psh3333/dacon-skku-bias-vlm-v5-grpo --repo-type model --private 2>/dev/null
hf upload psh3333/dacon-skku-bias-vlm-v5-grpo outputs/merged_v5 . --repo-type model 2>&1 | tail -2 || echo "HF WARN"

echo "=== 비교 (정직한 bbq_val_clean BA) ==="
echo "v4_BA: $(grep -aoE 'Balanced Accuracy: [0-9.]+' /workspace/master_v4.log | tail -1)"
echo "v5_BA: $(grep -aoE 'Balanced Accuracy: [0-9.]+' ${LOG} | tail -1)"
echo "=== MASTER v5 DONE $(date) ==="
touch /workspace/MASTER_v5_DONE
