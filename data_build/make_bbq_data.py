"""
BBQ(공개 벤치마크) → 대회 포맷 학습데이터 + 자체 검증셋 생성.

근거: 대회 데이터는 BBQ 멀티모달 버전(context/question/choices3/label, 'cannot be determined'류 기권 옵션).
BBQ(walledai/BBQ)는 동일 구조 + 라벨 有 → 실제 분포 학습 백본 + 라벨 있는 자체검증셋 확보.
※ 공개 벤치마크 사용은 규칙 허용. 평가셋 자체를 모방하는 게 아님(BBQ는 상위 공개 소스).

출력:
  - <out_dir>/bbq_train.json   : LLaMA-Factory 멀티모달 sharegpt (<image> + <think> 추론 + 정답)
  - <out_dir>/bbq_val.jsonl    : 자체 검증용 (context/question/answers/label/label_type/image_path) ← 단일 북극성 지표
  - <out_dir>/placeholder.jpg  : 이미지 자리(중립). 이미지는 '편향 함정'이라 보조 — 텍스트추론+기권이 핵심.

이미지 정책: BBQ는 텍스트 전용 → 중립 placeholder로 모달리티만 유지(모델은 context로 판단·기권 학습).
            추후 장면 이미지로 업그레이드 가능.
"""

import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, UNKNOWN_PATTERNS

SEED = 42
random.seed(SEED)


def is_unknown(opt):
    o = str(opt).lower()
    return any(p in o for p in UNKNOWN_PATTERNS)


def make_placeholder(path):
    """중립 회색 placeholder 이미지 생성 (의도적으로 정보 없음)."""
    from PIL import Image
    if not os.path.exists(path):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(path)


def reason_for(label_type):
    """짧은 <think> 추론 (0.5초 추론 예산 위해 1문장)."""
    if label_type == "ambiguous":
        return ("The context gives no clear evidence to identify a specific person; "
                "choosing one would rely on social stereotypes, so abstain.")
    return "The context states a clear fact that identifies the correct person."


def to_sharegpt(rec, image_path):
    """대회 포맷 dict → LLaMA-Factory 멀티모달 sharegpt 레코드 (<image> + <think> 타깃)."""
    user = "<image>" + USER_TEMPLATE.format(
        context=rec["context"], question=rec["question"],
        options_block=format_options(rec["answers"]))
    label = int(rec["label"])
    target = f"<think>\n{reason_for(rec['label_type'])}\n</think>\n{label}) {rec['answers'][label]}"
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
    ap.add_argument("--val_per_cat", type=int, default=120, help="카테고리별 검증 샘플 수")
    ap.add_argument("--max_train", type=int, default=0, help="0=전체, 양수=학습 상한")
    args = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)
    img = os.path.abspath(os.path.join(args.out_dir, "placeholder.jpg"))  # LLaMA-Factory가 cwd 무관하게 찾도록 절대경로
    make_placeholder(img)

    # LLaMA-Factory 데이터 등록 파일 (dataset_dir=out_dir로 바로 사용)
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump({"bbq_reasoning": {
            "file_name": "bbq_train.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
        }}, f, ensure_ascii=False, indent=2)

    d = load_dataset(args.hf)
    # 카테고리(split)별로 모아서 변환 + ambig/disambig 라벨링
    by_cat = {}
    for split in d.keys():
        for r in d[split]:
            answers = list(r["choices"])
            if len(answers) != 3:
                continue
            label = int(r["answer"])
            rec = {
                "context": r["context"], "question": r["question"],
                "answers": answers, "label": label,
                "label_type": "ambiguous" if is_unknown(answers[label]) else "disambiguated",
                "category": r.get("category", split),
            }
            by_cat.setdefault(split, []).append(rec)

    # 자체 검증셋: 카테고리별로 ambig/disambig 균형 있게 holdout
    train, val = [], []
    for cat, recs in by_cat.items():
        random.shuffle(recs)
        amb = [x for x in recs if x["label_type"] == "ambiguous"]
        dis = [x for x in recs if x["label_type"] == "disambiguated"]
        half = args.val_per_cat // 2
        val_cat = amb[:half] + dis[:half]
        val.extend(val_cat)
        rest = amb[half:] + dis[half:]
        train.extend(rest)

    random.shuffle(train)
    random.shuffle(val)
    if args.max_train > 0:
        train = train[:args.max_train]

    # 쓰기
    with open(os.path.join(args.out_dir, "bbq_train.json"), "w", encoding="utf-8") as f:
        json.dump([to_sharegpt(r, img) for r in train], f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "bbq_val.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            r2 = dict(r); r2["image_path"] = img
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

    n_amb = sum(r["label_type"] == "ambiguous" for r in train)
    print(f"TRAIN {len(train)} (ambiguous {n_amb}, disambiguated {len(train)-n_amb})")
    print(f"VAL   {len(val)} (ambiguous {sum(r['label_type']=='ambiguous' for r in val)})")
    print(f"OUT   {args.out_dir}/bbq_train.json , bbq_val.jsonl , placeholder.jpg")


if __name__ == "__main__":
    main()
