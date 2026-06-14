#!/usr/bin/env bash
# RunPod 턴키 셋업 — pod 안에서 한 번 붙여넣어 실행.
# 전제: 사용자가 직접 RunPod에 로그인 → GPU pod 대여(A100 80GB 또는 A6000 48GB) →
#       PyTorch 템플릿(CUDA 12.4) 선택 → 이 스크립트 실행.
#
# 이 스크립트가 하는 일: 의존성 설치 → (프로젝트 업로드 가정) → 학습 → 병합 → 추론.
# 데이터는 미리 data/ 에 올라와 있어야 함 (data/README.md 참조).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/dacon-bias-challenge}"
cd "$PROJECT_DIR"

echo "=== [1/5] 시스템/파이썬 확인 ==="
python --version
nvidia-smi || { echo "GPU 미감지 — pod GPU 설정 확인"; exit 1; }

echo "=== [2/5] 의존성 설치 ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [3/5] (선택) HuggingFace 로그인 ==="
# 비공개/게이트 모델 사용 시에만 필요. 토큰은 본인이 환경변수로 주입(코드에 하드코딩 금지).
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN"
else
  echo "HF_TOKEN 미설정 — 공개 모델만 사용 가정, 건너뜀"
fi

echo "=== [4/5] 데이터 점검 ==="
if [ ! -f "data/train_abstention.jsonl" ]; then
  echo "!! data/train_abstention.jsonl 없음 — 먼저 학습데이터를 구축/업로드하세요."
  echo "   (data_build/build_abstention_dataset.py 또는 로컬에서 만든 jsonl 업로드)"
  exit 1
fi

echo "=== [5/5] 학습 → 병합 → 추론 ==="
python train/train_vlm_qlora.py --config train/config.yaml
LATEST_CKPT="$(ls -d outputs/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo outputs)"
python train/merge_lora.py --adapter "$LATEST_CKPT" --out outputs/merged
if [ -f "data/test.csv" ]; then
  python inference/infer_vllm.py --model outputs/merged --test data/test.csv --out submissions/sub_v1.csv
  echo "제출 파일 생성: submissions/sub_v1.csv"
else
  echo "data/test.csv 없음 — 추론은 대회 평가데이터 확보 후 실행"
fi

echo "=== 완료 ==="
