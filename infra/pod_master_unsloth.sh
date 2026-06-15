#!/usr/bin/env bash
# Unsloth 풀 파이프라인 — v4(SFT)+v5(GRPO) 전부 Unsloth. 회원님 템플릿 기반, 일관성(셔플/dedup/누출/단일토큰) 유지.
# 학습=unsloth_env(FastVisionModel), 데이터/추론=lf311(datasets/transformers, Qwen3-VL merged 로드).
set -o pipefail
cd /workspace/dacon-bias-challenge
LOG=/workspace/master_unsloth.log
exec >> "$LOG" 2>&1
echo "=== MASTER UNSLOTH START $(date) ==="

echo "[0] Unsloth env 준비 대기..."
while ! test -f /workspace/UNSLOTH_SETUP_DONE; do sleep 30; done
grep -aq "UNSLOTH_IMPORT_OK" /workspace/unsloth_setup.log || { echo "ABORT: unsloth import 실패"; exit 1; }
while pgrep -f "llamafactory|train_grpo|sft_unsloth|grpo_unsloth|make_submission|baseline_eval" >/dev/null; do sleep 30; done
echo "[0] 준비됨 $(date)."

# 1) 데이터(lf311) — 셔플·dedup·누출제거·단일토큰 + val_ids SSOT
echo "[1] 클린 데이터 생성 (make_bbq_clean)"
( source /workspace/lf311/bin/activate; python data_build/make_bbq_clean.py ) || { echo "ABORT: 데이터"; exit 1; }
echo "[1b] 누출 검사 (test.csv 대조)"
( source /workspace/lf311/bin/activate; python data_build/contamination_check.py --out contamination_report.json 2>&1 | tail -3 ) || echo "WARN contam"

# 2) v4 = Unsloth SFT
echo "[2] v4 Unsloth SFT (FastVisionModel)"
source /workspace/unsloth_env/bin/activate; export WANDB_PROJECT=dacon-bias-challenge
rm -rf outputs/merged_v4
python train/unsloth/sft_unsloth.py --model Qwen/Qwen3-VL-8B-Instruct \
  --train data/bbq_v4_train.json --out outputs/merged_v4 --epochs 3 --rank 32 || { echo "ABORT: SFT"; exit 1; }

# 3) v4 평가 + 제출 (lf311/transformers)
echo "[3] v4 평가 스위트 + 제출"
( source /workspace/lf311/bin/activate
  bash infra/run_eval_suite.sh v4 outputs/merged_v4
  python inference/stress_eval.py --model outputs/merged_v4 --val data/bbq_val_clean.jsonl --limit 300 --max_new_tokens 8 --out data/v4_stress.json 2>&1 | tail -1
  python inference/make_submission.py --model outputs/merged_v4 --test data_build/test_full.csv \
    --images_dir /nonexistent --fallback_image data/placeholder.jpg --out submissions/submission_v4.csv --max_new_tokens 4 )

# 4) v5 = Unsloth GRPO (v4 위)
echo "[4] v5 Unsloth GRPO (fast_inference)"
source /workspace/unsloth_env/bin/activate
rm -rf outputs/merged_v5
python train/unsloth/grpo_unsloth.py --base outputs/merged_v4 --out outputs/merged_v5 \
  --val_ids data/val_ids.json --n 2000 --steps 300 --rank 32 --num_gen 4 || { echo "ABORT: GRPO(v5 실패, v4는 유효)"; touch /workspace/MASTER_UNSLOTH_DONE; exit 1; }

# 5) v5 평가 + 제출
echo "[5] v5 평가 스위트 + 제출"
( source /workspace/lf311/bin/activate
  bash infra/run_eval_suite.sh v5 outputs/merged_v5
  python inference/make_submission.py --model outputs/merged_v5 --test data_build/test_full.csv \
    --images_dir /nonexistent --fallback_image data/placeholder.jpg --out submissions/submission_v5.csv --max_new_tokens 4 )

# 6) 최종 선택 리포트 + HF
echo "[6] 모델 선택 리포트 + HF 업로드"
( source /workspace/lf311/bin/activate; export PATH="$HOME/.local/bin:$PATH"
  python inference/final_model_selection_report.py --tags v4,v5 2>&1 | tail -20
  for tag in v4 v5; do
    hf repo create psh3333/dacon-skku-bias-vlm-${tag}-unsloth --repo-type model --private 2>/dev/null
    hf upload psh3333/dacon-skku-bias-vlm-${tag}-unsloth outputs/merged_${tag} . --repo-type model 2>&1 | tail -1 || echo "HF WARN ${tag}"
  done )

echo "=== MASTER UNSLOTH DONE $(date) ==="
touch /workspace/MASTER_UNSLOTH_DONE
