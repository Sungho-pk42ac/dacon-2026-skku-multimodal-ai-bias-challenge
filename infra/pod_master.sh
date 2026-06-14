#!/usr/bin/env bash
# v2 풀사이클 자율 오케스트레이터 (overnight, pod nohup).
# 진행: 진짜추론 데이터 대기 → v1 제출 대기 → v2 학습 → 병합 → 자체검증 → submission_v2.csv → HF 배포
# 인자: $1 = 사이클 태그(v2/v3 등), $2 = dataset 키, $3 = epochs
set -o pipefail   # set -u 제거: ~/.bashrc source 시 unbound var로 죽는 문제 방지
TAG="${1:-v2}"
DSET="${2:-bbq_reasoning_real}"
EPOCHS="${3:-3}"
cd /workspace/dacon-bias-challenge
source ~/.bashrc 2>/dev/null
source /workspace/lf311/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT=dacon-bias-challenge
LOG=/workspace/master_${TAG}.log
exec >> "$LOG" 2>&1
echo "=== MASTER ${TAG} START $(date) ==="

echo "[1] 진짜추론 데이터 생성 대기..."
while pgrep -f make_bbq_reasoning >/dev/null; do sleep 30; done
test -f data/bbq_reason_train.json || { echo "ABORT: reason data 없음"; exit 1; }
echo "[1] done. train rows ~ $(grep -oc 'messages' data/bbq_reason_train.json 2>/dev/null || echo NA)"

echo "[2] v1 제출(있으면) 완료 대기..."
while pgrep -f make_submission.py >/dev/null; do sleep 30; done
echo "[2] GPU 확보."

echo "[3] v2 학습 (dataset=${DSET}, epochs=${EPOCHS})..."
rm -rf outputs/sft_${TAG}
llamafactory-cli train train/llamafactory/qwen25vl_lora_sft.yaml \
  dataset=${DSET} num_train_epochs=${EPOCHS} image_max_pixels=50176 cutoff_len=1024 \
  report_to=wandb run_name=sft_${TAG} output_dir=outputs/sft_${TAG} \
  overwrite_output_dir=true save_steps=300 logging_steps=10 || { echo "ABORT: train 실패"; exit 1; }

echo "[4] 병합..."
rm -rf outputs/merged_${TAG}
llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_name_or_path outputs/sft_${TAG} --template qwen2_vl --finetuning_type lora \
  --export_dir outputs/merged_${TAG} --export_size 5 --trust_remote_code true || { echo "ABORT: merge 실패"; exit 1; }

echo "[5] 자체검증 평가..."
python inference/baseline_eval.py --model outputs/merged_${TAG} --val data/bbq_val.jsonl --limit 0 \
  --out data/${TAG}_preds.csv

echo "[6] submission_${TAG}.csv 생성 (전체 8500)..."
python inference/make_submission.py --model outputs/merged_${TAG} --test data_build/test_full.csv \
  --images_dir /nonexistent --fallback_image data/placeholder.jpg --out submissions/submission_${TAG}.csv

echo "[7] HF 배포 (private)..."
hf repo create psh3333/dacon-skku-bias-vlm-${TAG} --repo-type model --private -y 2>/dev/null || \
  hf repo create psh3333/dacon-skku-bias-vlm-${TAG} --repo-type model --private 2>/dev/null || true
hf upload psh3333/dacon-skku-bias-vlm-${TAG} outputs/merged_${TAG} . --repo-type model 2>&1 | tail -2 || echo "WARN: HF 업로드 실패(모델은 로컬에 있음)"

echo "RESULT_BA: $(grep -aoE 'Balanced Accuracy: [0-9.]+' ${LOG} | tail -1)"
echo "SUB_ROWS: $(wc -l < submissions/submission_${TAG}.csv 2>/dev/null || echo NA)"
echo "=== MASTER ${TAG} DONE $(date) ==="
touch /workspace/MASTER_${TAG}_DONE
