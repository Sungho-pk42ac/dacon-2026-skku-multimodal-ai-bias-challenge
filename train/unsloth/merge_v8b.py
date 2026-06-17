"""
v8B 병합: base(merged_v4) + LoRA 어댑터(outputs/v8b_adapter) → outputs/merged_v8b (16bit).
디스크 절약을 위해 학습은 어댑터만 저장하므로, 평가/제출 전 1회 병합.
PeftModel.merge_and_unload (transformers, torch 2.6 호환 — eval 환경과 동일 경로).

실행: python train/unsloth/merge_v8b.py --base outputs/merged_v4 \
        --adapter outputs/v8b_adapter --out outputs/merged_v8b
"""
import argparse, os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--adapter", default="outputs/v8b_adapter")
    ap.add_argument("--out", default="outputs/merged_v8b")
    args = ap.parse_args()

    print(f"[MERGE] base={args.base} adapter={args.adapter} -> {args.out}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    print("MERGE_V8B_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
