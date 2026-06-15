"""
v4 직답(direct-answer) 학습데이터 생성 — 추론 속도 0.5초/샘플 규칙 준수용.

배경: make_bbq_data.py의 타깃은 `<think>이유</think>\n0) 답` 형태라 추론 시 이유문장을
      매번 생성 → ~0.88초/샘플(규칙 0.5초 초과). v4는 **답만**(`0) 답`) 출력하도록 학습해
      추론 디코딩 토큰을 최소화 → 속도 컴플라이언스 확보. 최종답은 여전히 LLM 생성물(규칙 #6 충족).

설계 원칙:
  - 기존 자체검증셋(bbq_val.jsonl)을 **그대로** 쓰기 위해 SEED/split을 make_bbq_data.py와 동일하게 맞춤
    → v2/v3/v4를 같은 검증셋에서 공정 비교. (val은 새로 쓰지 않음)
  - dataset_info.json은 **기존 키 보존**하며 bbq_direct만 병합(다른 데이터셋 키 덮어쓰지 않음).

출력:
  - <out_dir>/bbq_direct.json        : LLaMA-Factory sharegpt (<image> + 직답 타깃)
  - <out_dir>/dataset_info.json      : 기존 + {"bbq_direct": ...} 병합
"""

import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, UNKNOWN_PATTERNS

SEED = 42  # make_bbq_data.py와 동일 — 검증셋 holdout 일치 보장
random.seed(SEED)


def is_unknown(opt):
    o = str(opt).lower()
    return any(p in o for p in UNKNOWN_PATTERNS)


def to_sharegpt_direct(rec, image_path):
    """대회 포맷 dict → 직답 sharegpt 레코드 (<image> + '0) 답' 타깃, <think> 없음)."""
    user = "<image>" + USER_TEMPLATE.format(
        context=rec["context"], question=rec["question"],
        options_block=format_options(rec["answers"]))
    label = int(rec["label"])
    target = f"{label}) {rec['answers'][label]}"  # 직답: 선행 인덱스 → text_to_label이 즉시 파싱
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "images": [image_path],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--hf", default="walledai/BBQ")
    ap.add_argument("--val_per_cat", type=int, default=120, help="make_bbq_data.py와 동일해야 함")
    ap.add_argument("--max_train", type=int, default=0, help="0=전체")
    args = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)
    img = os.path.abspath(os.path.join(args.out_dir, "placeholder.jpg"))

    d = load_dataset(args.hf)
    by_cat = {}
    for split in d.keys():
        for r in d[split]:
            answers = list(r["choices"])
            if len(answers) != 3:
                continue
            label = int(r["answer"])
            by_cat.setdefault(split, []).append({
                "context": r["context"], "question": r["question"],
                "answers": answers, "label": label,
                "label_type": "ambiguous" if is_unknown(answers[label]) else "disambiguated",
            })

    # make_bbq_data.py와 동일한 split 로직 → 동일 val holdout 제외, 나머지를 train으로
    train = []
    for cat, recs in by_cat.items():
        random.shuffle(recs)
        amb = [x for x in recs if x["label_type"] == "ambiguous"]
        dis = [x for x in recs if x["label_type"] == "disambiguated"]
        half = args.val_per_cat // 2
        rest = amb[half:] + dis[half:]   # val(앞쪽 half+half)은 제외 → bbq_val.jsonl과 disjoint
        train.extend(rest)

    random.shuffle(train)
    if args.max_train > 0:
        train = train[:args.max_train]

    with open(os.path.join(args.out_dir, "bbq_direct.json"), "w", encoding="utf-8") as f:
        json.dump([to_sharegpt_direct(r, img) for r in train], f, ensure_ascii=False)

    # dataset_info.json 기존 키 보존 병합
    info_path = os.path.join(args.out_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    info["bbq_direct"] = {
        "file_name": "bbq_direct.json", "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    n_amb = sum(r["label_type"] == "ambiguous" for r in train)
    print(f"TRAIN(direct) {len(train)} (ambiguous {n_amb}, disambiguated {len(train)-n_amb})")
    print(f"OUT {args.out_dir}/bbq_direct.json + dataset_info.json(키 bbq_direct 병합)")


if __name__ == "__main__":
    main()
