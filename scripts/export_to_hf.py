"""
v8B Hugging Face 패키징/내보내기 (v8B Step 6). 재현가능한 공유 + 2차 평가용.

기본 동작: **로컬 패키징만** (업로드는 --push_to_hub 명시 시에만, 기본 private).
디스크 절약: 17GB merged 가중치는 복사하지 않고 LoRA 어댑터(작음)를 패키징.
            (base=merged_v4 + adapter로 재현. 최종추론 경로는 metadata.final_model_path에 기록.)

금지: DACON 테스트/비공개 데이터·원본 평가데이터·키/크레덴셜 업로드 금지.

산출(hf/v8b_package/):
  README.md, model_card.md, inference_example.py, metadata.json, requirements.txt, eval_summary.json
  adapter/ (LoRA 어댑터 사본; --adapter 제공 시)

실행(로컬 패키징):
  python scripts/export_to_hf.py --model outputs/merged_v8b --adapter outputs/v8b_adapter \
    --out hf/v8b_package --version v8B --submission submissions/submission_v8b.csv

실행(업로드):
  python scripts/export_to_hf.py --model outputs/merged_v8b --adapter outputs/v8b_adapter \
    --out hf/v8b_package --version v8B --submission submissions/submission_v8b.csv \
    --push_to_hub --hub_model_id psh3333/dacon-skku-bias-vlm-v8b --private
"""
import argparse, json, os, shutil, subprocess, sys


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/merged_v8b", help="최종추론용 merged 경로(메타 기록)")
    ap.add_argument("--adapter", default="outputs/v8b_adapter", help="패키징할 LoRA 어댑터 경로")
    ap.add_argument("--out", default="hf/v8b_package")
    ap.add_argument("--version", default="v8B")
    ap.add_argument("--submission", default="submissions/submission_v8b.csv")
    ap.add_argument("--eval_json", default="outputs/eval_results_v8b.json")
    ap.add_argument("--wandb_run", default="v8B_robust_grpo")
    ap.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_model_id", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 어댑터 사본(작음) + processor/generation_config ----
    adapter_copied = False
    if args.adapter and os.path.isdir(args.adapter):
        dst = os.path.join(args.out, "adapter")
        if os.path.abspath(dst) != os.path.abspath(args.adapter):
            shutil.copytree(args.adapter, dst, dirs_exist_ok=True)
        adapter_copied = True

    # ---- eval 요약 ----
    eval_summary = {}
    if os.path.exists(args.eval_json):
        try:
            allr = json.load(open(args.eval_json, encoding="utf-8"))
            eval_summary = allr.get("v8b", allr)
        except Exception:
            pass
    json.dump(eval_summary, open(os.path.join(args.out, "eval_summary.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- 학습 설정(있으면) ----
    train_cfg = {}
    if os.path.exists("outputs/v8b_train_log.json"):
        try:
            train_cfg = json.load(open("outputs/v8b_train_log.json", encoding="utf-8")).get("config", {})
        except Exception:
            pass

    # ---- metadata.json ----
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_ver = sh("python -c 'import torch;print(torch.__version__)'")
    cuda_ver = sh("nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | grep -oE '[0-9.]+'") or "12.4"
    meta = {
        "model_version": args.version,
        "base_model_name": args.base_model,
        "base_model_license": "Apache-2.0",
        "base_model_release_date": "2025-xx (Apache-2.0, 대회 허용 2026-06-01 이전 공개 확인 필요)",
        "base_model_url": f"https://huggingface.co/{args.base_model}",
        "training_method": "GRPO-only (no SFT)",
        "cold_start_sft_used": False,
        "long_reasoning_used": False,
        "dacon_eval_data_used_for_training": False,
        "dacon_test_pattern_mining_used": False,
        "training_data_sources": ["Elfsong/BBQ", "lighteval/siqa", "tau/commonsense_qa",
                                   "allenai/openbookqa", "allenai/ai2_arc"],
        "source_normalized_reward_used": True,
        "dynamic_sampling_used": True,
        "option_shuffle_reward_used": True,
        "git_commit_hash": sh("git rev-parse HEAD"),
        "python_version": py_ver,
        "cuda_version": cuda_ver,
        "torch_version": torch_ver,
        "final_model_path": args.model,
        "adapter_path": args.adapter if adapter_copied else None,
        "submission_csv_path": args.submission,
        "wandb_run_name": args.wandb_run,
        "train_config": train_cfg,
    }
    json.dump(meta, open(os.path.join(args.out, "metadata.json"), "w"), ensure_ascii=False, indent=2)

    # ---- requirements.txt (기준환경 torch 2.6) ----
    with open(os.path.join(args.out, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("torch==2.6.0\ntransformers>=4.49\naccelerate\npeft\npillow\npandas\n")

    # ---- inference_example.py (base+adapter 로드 예시; 외부 API 0) ----
    with open(os.path.join(args.out, "inference_example.py"), "w", encoding="utf-8") as f:
        f.write('''"""v8B 추론 예시 — base(merged_v4) + LoRA 어댑터. 최종답은 모델 생성 텍스트에서 파싱."""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE = "outputs/merged_v4"          # 또는 metadata.final_model_path(merged_v8b)
ADAPTER = "./adapter"               # 본 패키지의 LoRA 어댑터
SYS = ("You are a careful visual question answering assistant. Given an image, a context, "
       "a question, and three answer options, choose the single most appropriate option.\\n"
       "Do NOT rely on social stereotypes. Only commit when image and context give clear evidence. "
       "If not enough evidence, choose the 'cannot be determined' option.")

proc = AutoProcessor.from_pretrained(BASE); proc.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

def predict(image_path, context, question, answers):
    img = Image.open(image_path).convert("RGB")   # DACON 추론은 실제 이미지 사용
    opts = "\\n".join(f"{i}) {o}" for i, o in enumerate(answers))
    user = f"Context: {context}\\nQuestion: {question}\\nOptions:\\n{opts}\\nAnswer with the exact text of the single best option."
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": img}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[img], return_tensors="pt").to(model.device)
    g = model.generate(**inp, max_new_tokens=4, do_sample=False)
    out = proc.batch_decode([g[0][inp.input_ids.shape[1]:]], skip_special_tokens=True)[0]
    for ch in out.strip()[:3]:
        if ch in "012": return int(ch)
    return 0
''')

    # ---- model_card.md / README.md ----
    es = eval_summary or {}
    card = f"""# DACON SKKU Bias VLM — {args.version}

GRPO-only(콜드스타트 SFT 없음) 단일토큰 멀티모달 편향완화 모델. base: {args.base_model} (Apache-2.0).

## 학습 방법
- GRPO-only, LoRA(attention-only, MLP off), 단일토큰 출력(0/1/2).
- 보상 6종: answer / shuffle_consistency / abstain / source_normalized / format / length.
- 동적샘플링(오프라인 근사), 셔플짝 데이터증강으로 옵션순서 강건성 회복.
- DACON 평가셋·테스트 패턴 **미사용**. 공개데이터(BBQ/SIQA/CSQA/OBQA/ARC)만.

## 평가(요약)
- OOD acc: {es.get('ood_acc', 'TBD')} · option-shuffle consistency: {es.get('shuffle_consistency', 'TBD')}
- BBQ amb/dis: {es.get('bbq_amb_acc', 'TBD')}/{es.get('bbq_dis_acc', 'TBD')}

## 사용
`inference_example.py` 참고 (base merged_v4 + 본 어댑터). 최종 라벨은 모델 생성 텍스트에서 파싱.

## 라이선스/규칙 준수
- base Apache-2.0. 외부 API 추론 없음. 최종답=모델 텍스트(규칙기반 선택 아님).
- DACON 테스트/비공개 이미지·데이터는 학습/채굴/보상/프롬프트에 미사용.
"""
    open(os.path.join(args.out, "model_card.md"), "w", encoding="utf-8").write(card)
    open(os.path.join(args.out, "README.md"), "w", encoding="utf-8").write(card)

    print(f"HF_PACKAGE_DONE {args.out} adapter_copied={adapter_copied}", flush=True)

    # ---- 업로드(명시 시에만) ----
    if args.push_to_hub:
        if not args.hub_model_id:
            print("ERROR: --push_to_hub 에는 --hub_model_id 필요", flush=True); sys.exit(2)
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hub_model_id, private=args.private, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=args.out, repo_id=args.hub_model_id, repo_type="model")
        print(f"HF_UPLOAD_DONE {args.hub_model_id} private={args.private}", flush=True)


if __name__ == "__main__":
    main()
