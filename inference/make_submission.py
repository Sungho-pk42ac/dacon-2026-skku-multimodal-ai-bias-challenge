"""
제출 CSV 생성 (transformers 기반, 추론 전용 — 학습과 분리).

대회 test.csv(sample_id,image_path,context,question,answers) → sample_submission 형식(sample_id,label 0/1/2).
vLLM 미설치 환경에서도 동작(baseline_eval와 동일한 transformers 추론).
규칙: 외부 API 호출 0, 최종답은 LLM 생성 텍스트 → 3옵션 인덱스로 매핑.

실행: python inference/make_submission.py --model outputs/merged_v1 --test data_build/smoke_test_sample.csv \
        --images_dir data/test/images --fallback_image data/placeholder.jpg --out submissions/smoke_sub.csv
"""

import argparse
import ast
import csv
import json
import logging
import os
import sys

import pandas as pd
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_answers(raw):
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw)


def resolve_image(images_dir, image_path, fallback):
    base = os.path.basename(str(image_path))
    for cand in (os.path.join(images_dir, base),
                 os.path.join(images_dir, "..", str(image_path).lstrip("./"))):
        if os.path.exists(cand):
            return cand
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default="data/test/test.csv")
    ap.add_argument("--images_dir", default="data/test/images")
    ap.add_argument("--fallback_image", default="data/placeholder.jpg")
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    logger.info("모델 로드: %s", args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.model)

    df = pd.read_csv(args.test)
    logger.info("test 행수: %d", len(df))
    rows = []
    for i, r in df.iterrows():
        answers = parse_answers(r["answers"])
        img = Image.open(resolve_image(args.images_dir, r["image_path"], args.fallback_image)).convert("RGB")
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "text", "text": USER_TEMPLATE.format(
                    context=r["context"], question=r["question"], options_block=format_options(answers))},
                {"type": "image", "image": img}]},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        out = processor.batch_decode(
            [g[len(inp):] for inp, g in zip(inputs.input_ids, gen)], skip_special_tokens=True)[0]
        label = text_to_label(out, answers)
        rows.append({"sample_id": r["sample_id"], "label": label})
        if (i + 1) % 100 == 0:
            logger.info("%d/%d", i + 1, len(df))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label"])
        w.writeheader(); w.writerows(rows)
    logger.info("제출 파일 저장 → %s (%d행)", args.out, len(rows))


if __name__ == "__main__":
    main()
