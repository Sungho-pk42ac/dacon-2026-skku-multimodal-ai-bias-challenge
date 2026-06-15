#!/usr/bin/env bash
# v4 오케스트레이터 (회의 확정 클린버전) — Qwen3-VL-8B + 누출제거 + 단일토큰 직답 + 0.5초 실측.
# 흐름: v3제출완료/GPU대기 → 누출제거 데이터 → SFT(Qwen3-VL nothink) → 병합 → 정직한 BA → 속도실측 → 제출CSV → HF.
set -o pipefail
cd /workspace/dacon-bias-challenge
LOG=/workspace/master_v4.log
exec >> "$LOG" 2>&1
echo "=== MASTER v4 (Qwen3-VL clean) START $(date) ==="

echo "[0] v3 제출 완료 + GPU 확보 대기..."
while ! test -f /workspace/V3_SUBMISSION_DONE; do sleep 30; done
while pgrep -f "make_submission|baseline_eval|llamafactory|train_grpo" >/dev/null; do sleep 30; done
echo "[0] GPU 확보됨 $(date)."

source ~/.bashrc 2>/dev/null
source /workspace/lf311/bin/activate
export PATH="$HOME/.local/bin:$PATH"; export WANDB_PROJECT=dacon-bias-challenge

echo "[1] 누출제거 클린 데이터 생성 (sha1 sample_id + val SSOT + 단일토큰)..."
python data_build/make_bbq_clean.py || { echo "ABORT: 데이터생성 실패"; exit 1; }

echo "[2] v4 SFT (Qwen3-VL-8B-Instruct, qwen3_vl_nothink, dataset=bbq_v4)..."
rm -rf outputs/sft_v4
llamafactory-cli train train/llamafactory/qwen3vl_lora_sft.yaml || { echo "ABORT: 학습 실패"; exit 1; }

echo "[3] 병합 → merged_v4..."
rm -rf outputs/merged_v4
llamafactory-cli export --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_name_or_path outputs/sft_v4 --template qwen3_vl_nothink --finetuning_type lora \
  --export_dir outputs/merged_v4 --export_size 5 --trust_remote_code true || { echo "ABORT: 병합 실패"; exit 1; }

echo "[4] 정직한 BA (누출제거 bbq_val_clean.jsonl, 단일토큰이라 max_new_tokens=8)..."
python inference/baseline_eval.py --model outputs/merged_v4 --val data/bbq_val_clean.jsonl --limit 0 \
  --max_new_tokens 8 --out data/v4_preds.csv

echo "[5] 속도 실측 (200샘플, max_new_tokens=4) — 0.5초 하드게이트 검증..."
python - <<'PYEOF'
import time, ast, json, sys, os
sys.path.append(os.path.abspath("."))
import pandas as pd, torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
M="outputs/merged_v4"
model=AutoModelForImageTextToText.from_pretrained(M,torch_dtype=torch.bfloat16,device_map="auto").eval()
proc=AutoProcessor.from_pretrained(M)
df=pd.read_csv("data_build/test_full.csv").head(200)
ph=Image.open("data/placeholder.jpg").convert("RGB")
def ans(r):
    try: return json.loads(r) if isinstance(r,str) else r
    except: return ast.literal_eval(r)
def run(r):
    a=ans(r["answers"]); msgs=[{"role":"system","content":[{"type":"text","text":SYSTEM_MESSAGE}]},{"role":"user","content":[{"type":"text","text":USER_TEMPLATE.format(context=r["context"],question=r["question"],options_block=format_options(a))},{"type":"image","image":ph}]}]
    t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(msgs)
    inp=proc(text=[t],images=im,videos=vi,padding=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): model.generate(**inp,max_new_tokens=4,do_sample=False)
for _,r in df.head(3).iterrows(): run(r)   # 워밍업
t0=time.time()
for _,r in df.iterrows(): run(r)
el=time.time()-t0
ok="PASS ✅" if el/len(df)<=0.5 else "FAIL ❌ (>0.5s)"
print(f"SPEED_PER_SAMPLE: {el/len(df):.3f} s/sample [{ok}]  8500환산 {el/len(df)*8500/60:.1f}분 [규칙 70분]")
PYEOF

echo "[6] submission_v4.csv (전체 8500, max_new_tokens=4)..."
python inference/make_submission.py --model outputs/merged_v4 --test data_build/test_full.csv \
  --images_dir /nonexistent --fallback_image data/placeholder.jpg \
  --out submissions/submission_v4.csv --max_new_tokens 4

echo "[7] v4 HF 업로드 (private)..."
hf repo create psh3333/dacon-skku-bias-vlm-v4 --repo-type model --private 2>/dev/null
hf upload psh3333/dacon-skku-bias-vlm-v4 outputs/merged_v4 . --repo-type model 2>&1 | tail -2 || echo "HF WARN"

echo "RESULT_V4_BA: $(grep -aoE 'Balanced Accuracy: [0-9.]+' ${LOG} | tail -1)"
echo "RESULT_V4_SPEED: $(grep -a 'SPEED_PER_SAMPLE' ${LOG} | tail -1)"
echo "=== MASTER v4 DONE $(date) ==="
touch /workspace/MASTER_v4_DONE
