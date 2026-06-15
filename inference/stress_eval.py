"""
스트레스 검증 — "Public 벤치 과적합 안 함"을 *증명*하는 자체 검증 (운영진 공지 대응).

운영진 공지(2026-06-15): Private는 운영진 자체제작 샘플. Public(오픈벤치) 과적합 모델은 Private서 미끄러짐.
"신빙성 있는 검증 과정을 어떻게 구성하는지"가 평가 대상 → 본 스크립트가 그 증거.

측정:
  1) **위치 강건성**: 각 문항을 선택지 6가지 순열(3!)로 평가 → 위치 바꿔도 같은 '의미'를 고르는가.
     - 위치에 과적합된 모델 = 순열 따라 답이 흔들림(consistency↓).
  2) **순열 평균 BA**: 위치 편향 제거된 진짜 정확도.
  3) **카테고리별 BA**: 다양한 편향축에 일관적인가(약한 축 진단).

실행: python inference/stress_eval.py --model outputs/merged_v4 --val data/bbq_val_clean.jsonl --limit 300 --max_new_tokens 8
"""

import argparse
import itertools
import json
import logging
import os
import sys
from collections import defaultdict

import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", default="data/bbq_val_clean.jsonl")
    ap.add_argument("--limit", type=int, default=300, help="문항 수(각 문항 ×6 순열). 0=전체")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default="data/stress_report.json")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    rows = []
    with open(args.val, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    logger.info("문항 %d × 6순열 = %d 추론", len(rows), len(rows) * 6)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.model)

    def predict(context, question, answers, image_path):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "text", "text": USER_TEMPLATE.format(
                    context=context, question=question, options_block=format_options(answers))},
                {"type": "image", "image": Image.open(image_path).convert("RGB")}]},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inputs = processor(text=[text], images=imgs, videos=vids, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        out = processor.batch_decode(
            [g[len(i):] for i, g in zip(inputs.input_ids, gen)], skip_special_tokens=True)[0]
        return text_to_label(out, answers)

    perms = list(itertools.permutations([0, 1, 2]))  # 6가지
    consistent = 0
    hit = {"ambiguous": 0, "disambiguated": 0}
    tot = {"ambiguous": 0, "disambiguated": 0}
    cat_hit = defaultdict(lambda: {"ambiguous": 0, "disambiguated": 0})
    cat_tot = defaultdict(lambda: {"ambiguous": 0, "disambiguated": 0})

    for idx, r in enumerate(rows):
        A = r["answers"]; gold = int(r["label"]); gold_text = A[gold]
        lt = r["label_type"]; cat = r.get("category", "?")
        chosen_texts = []
        for perm in perms:
            pa = [A[perm[i]] for i in range(3)]      # 순열된 선택지
            gold_pos = perm.index(gold)               # 이 순열에서 정답 위치
            pred = predict(r["context"], r["question"], pa, r["image_path"])
            chosen_texts.append(pa[pred] if 0 <= pred < 3 else None)
            tot[lt] += 1; cat_tot[cat][lt] += 1
            if pred == gold_pos:
                hit[lt] += 1; cat_hit[cat][lt] += 1
        # 위치 일관성: 6순열에서 항상 같은 '의미'(텍스트)를 골랐나
        if len(set(chosen_texts)) == 1:
            consistent += 1
        if (idx + 1) % 50 == 0:
            logger.info("%d/%d 문항", idx + 1, len(rows))

    acc_a = hit["ambiguous"] / max(tot["ambiguous"], 1)
    acc_d = hit["disambiguated"] / max(tot["disambiguated"], 1)
    ba = (acc_a + acc_d) / 2
    consistency = consistent / max(len(rows), 1)
    cat_ba = {}
    for cat in cat_tot:
        a = cat_hit[cat]["ambiguous"] / max(cat_tot[cat]["ambiguous"], 1)
        dd = cat_hit[cat]["disambiguated"] / max(cat_tot[cat]["disambiguated"], 1)
        cat_ba[cat] = round((a + dd) / 2, 4)

    report = {
        "model": args.model, "n_items": len(rows), "perms_per_item": 6,
        "position_robust_BA": round(ba, 4),
        "ambiguous_acc": round(acc_a, 4), "disambiguated_acc": round(acc_d, 4),
        "position_consistency": round(consistency, 4),   # 1.0=위치에 전혀 안 휘둘림
        "category_BA": dict(sorted(cat_ba.items(), key=lambda x: x[1])),
        "worst_category": min(cat_ba.items(), key=lambda x: x[1]) if cat_ba else None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("=== 위치강건 BA: %.4f | 위치일관성: %.4f (1.0=완벽) ===", ba, consistency)
    logger.info("=== 약한 카테고리: %s ===", report["worst_category"])
    print(f"STRESS position_robust_BA={ba:.4f} consistency={consistency:.4f} "
          f"worst_cat={report['worst_category']}")


if __name__ == "__main__":
    main()
