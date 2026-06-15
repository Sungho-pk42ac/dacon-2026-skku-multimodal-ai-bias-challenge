"""
자체 검증셋(BBQ holdout)에 대한 Balanced Accuracy 측정.
학습 전 베이스 모델 zero-shot 기준점, 학습 후 비교 모두 사용.

지표: Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated)/2  ← 대회와 동일, 단일 북극성.
입력: data/bbq_val.jsonl (context/question/answers/label/label_type/image_path)
실행(py3.11 lf311 env 권장): python inference/baseline_eval.py --model Qwen/Qwen2.5-VL-7B-Instruct --limit 0
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_val(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit and limit > 0 else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--val", default="data/bbq_val.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--out", default="data/baseline_preds.csv")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    logger.info("모델 로드: %s", args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    rows = load_val(args.val, args.limit)
    logger.info("검증 샘플: %d", len(rows))

    from collections import defaultdict
    hit = {"ambiguous": 0, "disambiguated": 0}
    tot = {"ambiguous": 0, "disambiguated": 0}
    cat_hit = defaultdict(lambda: {"ambiguous": 0, "disambiguated": 0})  # 카테고리별 분해(Hidden 일반화 대리)
    cat_tot = defaultdict(lambda: {"ambiguous": 0, "disambiguated": 0})
    out_rows = []
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "text", "text": USER_TEMPLATE.format(
                    context=r["context"], question=r["question"],
                    options_block=format_options(r["answers"]))},
                {"type": "image", "image": Image.open(r["image_path"]).convert("RGB")},
            ]},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        out = processor.batch_decode(
            [g[len(i):] for i, g in zip(inputs.input_ids, gen)],
            skip_special_tokens=True)[0]
        pred = text_to_label(out, r["answers"])
        lt = r["label_type"]
        cat = r.get("category", "?")
        tot[lt] += 1
        cat_tot[cat][lt] += 1
        if pred == int(r["label"]):
            hit[lt] += 1
            cat_hit[cat][lt] += 1
        out_rows.append({"pred": pred, "gold": r["label"], "label_type": lt, "category": cat})
        if (i + 1) % 50 == 0:
            logger.info("%d/%d done", i + 1, len(rows))

    acc_a = hit["ambiguous"] / max(tot["ambiguous"], 1)
    acc_d = hit["disambiguated"] / max(tot["disambiguated"], 1)
    ba = (acc_a + acc_d) / 2
    elapsed = time.time() - t0
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pred", "gold", "label_type", "category"])
        w.writeheader(); w.writerows(out_rows)
    logger.info("=== ambiguous Acc: %.4f (n=%d) ===", acc_a, tot["ambiguous"])
    logger.info("=== disambiguated Acc: %.4f (n=%d) ===", acc_d, tot["disambiguated"])
    logger.info("=== Balanced Accuracy: %.4f ===", ba)
    logger.info("샘플당 %.3fs (A6000 기준 0.5s 목표)", elapsed / max(len(rows), 1))

    # 카테고리별 분해 BA — Hidden 일반화 약점 진단(가장 낮은 카테고리가 위험)
    logger.info("--- 카테고리별 Balanced Accuracy (낮은 순) ---")
    cat_ba = {}
    for cat in cat_tot:
        a = cat_hit[cat]["ambiguous"] / max(cat_tot[cat]["ambiguous"], 1)
        dd = cat_hit[cat]["disambiguated"] / max(cat_tot[cat]["disambiguated"], 1)
        cat_ba[cat] = (a + dd) / 2
    for cat, v in sorted(cat_ba.items(), key=lambda x: x[1]):
        n = cat_tot[cat]["ambiguous"] + cat_tot[cat]["disambiguated"]
        logger.info("  %-22s BA=%.4f (n=%d)", cat, v, n)
    worst = min(cat_ba.items(), key=lambda x: x[1]) if cat_ba else ("?", 0)
    print(f"RESULT BA={ba:.4f} amb={acc_a:.4f} dis={acc_d:.4f} per_sample={elapsed/max(len(rows),1):.3f}s "
          f"worst_cat={worst[0]}:{worst[1]:.4f}")


if __name__ == "__main__":
    main()
