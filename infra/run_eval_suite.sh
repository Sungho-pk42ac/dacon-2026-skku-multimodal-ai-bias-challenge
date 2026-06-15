#!/usr/bin/env bash
# Tier1 평가 스위트 — merged 모델 1개에 대해 전체 자체검증 실행(수동/재현 가능).
# 사용: bash infra/run_eval_suite.sh <tag> <merged_model_path>
#   예: bash infra/run_eval_suite.sh v4 outputs/merged_v4
set -o pipefail
cd /workspace/dacon-bias-challenge
TAG="${1:?tag 필요(v4/v5)}"
MODEL="${2:-outputs/merged_${TAG}}"
source /workspace/lf311/bin/activate 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
echo "=== EVAL SUITE [$TAG] model=$MODEL $(date) ==="

echo "[1] 정직한 BA + 카테고리 (baseline_eval)"
python inference/baseline_eval.py --model "$MODEL" --val data/bbq_val_clean.jsonl --limit 0 \
  --max_new_tokens 8 --out data/${TAG}_preds.csv 2>&1 | grep -aE "RESULT|Balanced|카테고리" | tail -15

echo "[2] 위치 스트레스 (6순열 일관성)"
python inference/stress_eval.py --model "$MODEL" --val data/bbq_val_clean.jsonl \
  --limit 300 --max_new_tokens 8 --out data/${TAG}_stress.json 2>&1 | grep -a "STRESS" | tail -1

echo "[3] Counterfactual 스트레스 (사회속성 swap)"
python inference/counterfactual_stress.py --model "$MODEL" --val data/bbq_val_clean.jsonl \
  --limit 300 --max_new_tokens 8 --out data/${TAG}_cf.json 2>&1 | grep -a "CF " | tail -1

echo "=== EVAL SUITE [$TAG] DONE $(date) ==="
echo "산출물: data/${TAG}_preds.csv, data/${TAG}_stress.json, data/${TAG}_cf.json"
