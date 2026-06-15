"""
v4 클린 데이터 빌더 — 회의 BLOCKER(val 누출) 해결 + 단일토큰 직답 타깃.

회의(data-risk) 진단: 기존 make_bbq_data.py(카테고리 셔플)와 make_bbq_reasoning.py(글로벌 셔플)는
분할축·RNG 소비순서가 달라 같은 SEED로도 train/val disjoint 보장이 안 됐고, sample_id가 없어
중복 검출조차 불가 → 자체검증 BA가 부풀려짐(INFLATED).

처방(SSOT 방식):
  1) 모든 BBQ 레코드에 sha1 내용해시 sample_id 부여(context|question|answers).
  2) sample_id로 중복 제거(동일 문항이 train/val 양쪽에 새는 것 원천 차단).
  3) 균형 val 추출 → val_ids.json(단일 진실원천) 기록.
  4) train = 전체에서 val_ids를 **전역 배제** + assert 가드(교집합 0 보장).
  5) 타깃 = 단일 토큰 "0"/"1"/"2" (회의 U2: 12토큰 잘림+속도 동시 해결).

출력(data/):
  - val_ids.json          : 검증셋 sample_id 목록 (SSOT — 모든 생성기가 이걸 배제해야 함)
  - bbq_val_clean.jsonl   : 자체검증셋 (context/question/answers/label/label_type/image_path/sample_id)
  - bbq_v4_train.json     : LLaMA-Factory sharegpt (<image> + 단일토큰 직답)
  - dataset_info.json     : 기존 키 보존 + bbq_v4 병합
"""

import argparse
import hashlib
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


def sample_id(context, question, answers):
    """내용 기반 안정적 ID — train/val 중복 검출·배제의 키."""
    key = "|".join([str(context).strip(), str(question).strip(), "||".join(str(a).strip() for a in answers)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def make_placeholder(path):
    from PIL import Image
    if not os.path.exists(path):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(path)


def to_sharegpt_single_token(rec, image_path):
    """단일토큰 직답 타깃: assistant = '0'/'1'/'2' (회의 U2). 최종답은 LLM 생성물(규칙 충족)."""
    user = "<image>" + USER_TEMPLATE.format(
        context=rec["context"], question=rec["question"],
        options_block=format_options(rec["answers"]))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(int(rec["label"]))},  # 단일 토큰
        ],
        "images": [image_path],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--hf", default="walledai/BBQ")
    ap.add_argument("--val_per_cat", type=int, default=120)
    ap.add_argument("--max_train", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)
    img = os.path.abspath(os.path.join(args.out_dir, "placeholder.jpg"))
    make_placeholder(img)

    d = load_dataset(args.hf)

    # 1) 전체 레코드 + sample_id, 2) 중복 제거
    seen = {}
    for split in d.keys():
        for r in d[split]:
            answers = list(r["choices"])
            if len(answers) != 3:
                continue
            sid = sample_id(r["context"], r["question"], answers)
            if sid in seen:
                continue  # 중복 문항 1회만
            label = int(r["answer"])
            seen[sid] = {
                "sample_id": sid, "context": r["context"], "question": r["question"],
                "answers": answers, "label": label,
                "label_type": "ambiguous" if is_unknown(answers[label]) else "disambiguated",
                "category": r.get("category", split),
            }
    allrecs = list(seen.values())

    # 3) 카테고리별 균형 val 추출
    by_cat = {}
    for rec in allrecs:
        by_cat.setdefault(rec["category"], []).append(rec)
    val = []
    for cat, recs in by_cat.items():
        random.shuffle(recs)
        amb = [x for x in recs if x["label_type"] == "ambiguous"]
        dis = [x for x in recs if x["label_type"] == "disambiguated"]
        half = args.val_per_cat // 2
        val.extend(amb[:half] + dis[:half])
    val_ids = {r["sample_id"] for r in val}

    # 4) train = 전체 − val_ids (전역 배제) + assert 가드
    train = [r for r in allrecs if r["sample_id"] not in val_ids]
    overlap = val_ids & {r["sample_id"] for r in train}
    assert not overlap, f"LEAKAGE! train∩val = {len(overlap)}건"  # 누출 0 보장

    random.shuffle(train)
    if args.max_train > 0:
        train = train[:args.max_train]

    # 5) 쓰기 — SSOT + 검증셋 + 학습셋
    with open(os.path.join(args.out_dir, "val_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(val_ids), f, indent=2)
    with open(os.path.join(args.out_dir, "bbq_val_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            r2 = dict(r); r2["image_path"] = img
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "bbq_v4_train.json"), "w", encoding="utf-8") as f:
        json.dump([to_sharegpt_single_token(r, img) for r in train], f, ensure_ascii=False)

    # dataset_info 병합(기존 키 보존)
    info_path = os.path.join(args.out_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    info["bbq_v4"] = {
        "file_name": "bbq_v4_train.json", "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    n_amb_t = sum(r["label_type"] == "ambiguous" for r in train)
    n_amb_v = sum(r["label_type"] == "ambiguous" for r in val)
    print(f"[CLEAN] 전체 고유문항 {len(allrecs)} (중복제거 후)")
    print(f"[CLEAN] VAL {len(val)} (amb {n_amb_v}/dis {len(val)-n_amb_v}) → val_ids.json + bbq_val_clean.jsonl")
    print(f"[CLEAN] TRAIN {len(train)} (amb {n_amb_t}/dis {len(train)-n_amb_t}) → bbq_v4_train.json [단일토큰 타깃]")
    print(f"[CLEAN] train∩val 교집합 = {len(overlap)}건 (0이어야 정상) ✅")


if __name__ == "__main__":
    main()
