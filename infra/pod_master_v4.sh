#!/usr/bin/env bash
# v4 오케스트레이터 — 속도 컴플라이언스(0.5초/샘플) 버전.
# 흐름: v3 제출완료/GPU대기 → 직답데이터 생성 → SFT(bbq_direct) → 병합 → 자체검증 → 속도측정 → 제출CSV(max_new_tokens=12) → HF.
# 핵심: <think> 없는 직답 학습 + 짧은 디코딩 → 추론 ~0.2초/샘플 목표.
set -o pipefail
cd /workspace/dacon-bias-challenge
LOG=/workspace/master_v4.log
exec >> "$LOG" 2>&1
echo "=== MASTER v4 START $(date) ==="

echo "[0] v3 제출 완료 + GPU 확보 대기..."
while ! test -f /workspace/V3_SUBMISSION_DONE; do sleep 30; done
while pgrep -f "make_submission|baseline_eval|llamafactory|train_grpo" >/dev/null; do sleep 30; done
echo "[0] GPU 확보됨 $(date)."

source ~/.bashrc 2>/dev/null
source /workspace/lf311/bin/activate
export PATH="$HOME/.local/bin:$PATH"; export WANDB_PROJECT=dacon-bias-challenge

echo "[1] 직답 데이터 생성 (bbq_direct)..."
python data_build/make_bbq_direct.py || { echo "ABORT: 데이터생성 실패"; exit 1; }

echo "[2] v4 SFT (dataset=bbq_direct, 3ep)..."
rm -rf outputs/sft_v4
llamafactory-cli train train/llamafactory/qwen25vl_lora_sft.yaml \
  dataset=bbq_direct num_train_epochs=3 image_max_pixels=50176 cutoff_len=1024 \
  report_to=wandb run_name=sft_v4 output_dir=outputs/sft_v4 \
  overwrite_output_dir=true save_steps=300 logging_steps=10 || { echo "ABORT: 학습 실패"; exit 1; }

echo "[3] 병합..."
rm -rf outputs/merged_v4
llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_name_or_path outputs/sft_v4 --template qwen2_vl --finetuning_type lora \
  --export_dir outputs/merged_v4 --export_size 5 --trust_remote_code true || { echo "ABORT: 병합 실패"; exit 1; }

echo "[4] 자체검증 BA (v2/v3와 동일 bbq_val.jsonl)..."
python inference/baseline_eval.py --model outputs/merged_v4 --val data/bbq_val.jsonl --limit 0 \
  --out data/v4_preds.csv

echo "[5] 속도 측정 (200샘플, max_new_tokens=12)..."
python - <<'PYEOF'
import time, ast, json, sys, os
sys.path.append(os.path.abspath("."))
import pandas as pd, torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
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
# 워밍업
for _,r in df.head(3).iterrows():
    a=ans(r["answers"]); msgs=[{"role":"system","content":[{"type":"text","text":SYSTEM_MESSAGE}]},{"role":"user","content":[{"type":"text","text":USER_TEMPLATE.format(context=r["context"],question=r["question"],options_block=format_options(a))},{"type":"image","image":ph}]}]
    t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(msgs)
    inp=proc(text=[t],images=im,videos=vi,padding=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): model.generate(**inp,max_new_tokens=12,do_sample=False)
t0=time.time()
for _,r in df.iterrows():
    a=ans(r["answers"]); msgs=[{"role":"system","content":[{"type":"text","text":SYSTEM_MESSAGE}]},{"role":"user","content":[{"type":"text","text":USER_TEMPLATE.format(context=r["context"],question=r["question"],options_block=format_options(a))},{"type":"image","image":ph}]}]
    t=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); im,vi=process_vision_info(msgs)
    inp=proc(text=[t],images=im,videos=vi,padding=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): g=model.generate(**inp,max_new_tokens=12,do_sample=False)
el=time.time()-t0
print(f"SPEED_PER_SAMPLE: {el/len(df):.3f} s/sample  (200샘플 {el:.1f}s) -> 8500개 환산 {el/len(df)*8500/60:.1f}분  [규칙 70분/0.5초]")
PYEOF

echo "[6] submission_v4.csv 생성 (전체 8500, max_new_tokens=12)..."
python inference/make_submission.py --model outputs/merged_v4 --test data_build/test_full.csv \
  --images_dir /nonexistent --fallback_image data/placeholder.jpg \
  --out submissions/submission_v4.csv --max_new_tokens 12

echo "[7] v4 HF 업로드 (private)..."
hf repo create psh3333/dacon-skku-bias-vlm-v4 --repo-type model --private 2>/dev/null
hf upload psh3333/dacon-skku-bias-vlm-v4 outputs/merged_v4 . --repo-type model 2>&1 | tail -2 || echo "HF WARN"

echo "RESULT_V4_BA: $(grep -aoE 'Balanced Accuracy: [0-9.]+' ${LOG} | tail -1)"
echo "RESULT_V4_SPEED: $(grep -a 'SPEED_PER_SAMPLE' ${LOG} | tail -1)"
echo "=== MASTER v4 DONE $(date) ==="
touch /workspace/MASTER_v4_DONE
