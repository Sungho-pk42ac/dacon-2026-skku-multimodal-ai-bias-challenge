"""
v4 클린 데이터 빌더 (철저화판) — 회의 BLOCKER(val 누출) + 위치편향 + 시나리오누출 해결.

회의 진단(data-risk) + 데이터 철저화(2026-06-15):
  1) val 누출: split 로직 차이로 train/val disjoint 미보장 → BA 부풀려짐(INFLATED).
  2) 위치편향: BBQ는 'unknown' 옵션을 항상 마지막(idx 2)에 둠 → 모델이 "애매하면 2번"이라는
     **위치 규칙**을 외움(의미 아님). Hidden이 위치를 바꾸면 무너짐.
  3) 시나리오누출: 같은 상황의 ambiguous/disambiguated 버전이 train/val로 쪼개질 수 있음.

처방:
  1) sha1 sample_id → val_ids.json(SSOT) → 전역 배제 + assert.
  2) **옵션 순서 셔플**(라벨 재매핑) — train/val 모두. 모델이 위치 아닌 '기권의 의미'를 학습.
  3) **시나리오 단위 dedup**(scenario_id = 질문+정렬된 선택지) → 한 시나리오는 train/val 한쪽에만.
  4) **품질 필터**: unknown 옵션 정확히 1개 + 라벨 유효성.
  5) 단일토큰 직답 타깃("0"/"1"/"2").

출력(data/): val_ids.json · bbq_val_clean.jsonl · bbq_v4_train.json · dataset_info.json(bbq_v4 병합)
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, UNKNOWN_PATTERNS

SEED = 42
random.seed(SEED)


def is_unknown(opt):
    o = str(opt).lower()
    return any(p in o for p in UNKNOWN_PATTERNS)


def sample_id(context, question, answers):
    key = "|".join([str(context).strip(), str(question).strip(), "||".join(str(a).strip() for a in answers)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def scenario_id(question, answers):
    """같은 상황 식별 — 질문+정렬 선택지(amb/dis는 context만 다름)로 묶어 train/val 분리."""
    key = str(question).strip() + "||" + "||".join(sorted(str(a).strip() for a in answers))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def shuffle_opts(answers, label, seed_str):
    """선택지 순서를 결정적으로 셔플 + 라벨 재매핑 (위치편향 제거)."""
    rng = random.Random(seed_str)            # sample_id 기반 → 재현 가능
    order = [0, 1, 2]
    rng.shuffle(order)
    new_answers = [answers[i] for i in order]
    new_label = order.index(label)           # 정답(old label)의 새 위치
    return new_answers, new_label


def make_placeholder(path):
    from PIL import Image
    if not os.path.exists(path):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(path)


def to_sharegpt_single_token(rec, image_path):
    """단일토큰 직답 타깃: assistant = '0'/'1'/'2' (셔플된 라벨). 최종답은 LLM 생성물."""
    user = "<image>" + USER_TEMPLATE.format(
        context=rec["context"], question=rec["question"],
        options_block=format_options(rec["answers"]))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(int(rec["label"]))},
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

    # 1) 중복제거 + 품질필터 + sample_id/scenario_id
    seen = {}
    n_malformed = 0
    for split in d.keys():
        for r in d[split]:
            answers = list(r["choices"])
            if len(answers) != 3:
                n_malformed += 1; continue
            if sum(is_unknown(a) for a in answers) != 1:   # 품질: unknown 정확히 1개
                n_malformed += 1; continue
            label = int(r["answer"])
            if label not in (0, 1, 2):
                n_malformed += 1; continue
            sid = sample_id(r["context"], r["question"], answers)
            if sid in seen:
                continue
            seen[sid] = {
                "sample_id": sid,
                "scenario": scenario_id(r["question"], answers),
                "context": r["context"], "question": r["question"],
                "answers": answers, "label": label,
                "label_type": "ambiguous" if is_unknown(answers[label]) else "disambiguated",
                "category": split,
            }
    allrecs = list(seen.values())

    # 2) 카테고리별 → 시나리오 단위로 train/val 분리 (시나리오 atomic)
    by_cat = defaultdict(lambda: defaultdict(list))
    for rec in allrecs:
        by_cat[rec["category"]][rec["scenario"]].append(rec)

    val, train = [], []
    for cat, scen_map in by_cat.items():
        scen_keys = list(scen_map.keys())
        random.shuffle(scen_keys)
        cnt = 0
        for sk in scen_keys:
            recs = scen_map[sk]
            if cnt < args.val_per_cat:
                val.extend(recs); cnt += len(recs)   # 시나리오 통째로 val
            else:
                train.extend(recs)

    # 시나리오 누출 0 보장
    val_scens = {r["scenario"] for r in val}
    leak = [r for r in train if r["scenario"] in val_scens]
    assert not leak, f"SCENARIO LEAK! {len(leak)}건"
    val_ids = {r["sample_id"] for r in val}

    random.shuffle(train)
    if args.max_train > 0:
        train = train[:args.max_train]

    # 3) 옵션 셔플(train/val 모두) — 위치편향 제거
    def apply_shuffle(rec):
        na, nl = shuffle_opts(rec["answers"], rec["label"], rec["sample_id"])
        rec2 = dict(rec); rec2["answers"] = na; rec2["label"] = nl
        return rec2  # label_type은 정답 텍스트 기준이라 셔플 무관(유지)
    train = [apply_shuffle(r) for r in train]
    val = [apply_shuffle(r) for r in val]

    # 4) 쓰기
    with open(os.path.join(args.out_dir, "val_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(val_ids), f, indent=2)
    with open(os.path.join(args.out_dir, "bbq_val_clean.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            r2 = dict(r); r2["image_path"] = img
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "bbq_v4_train.json"), "w", encoding="utf-8") as f:
        json.dump([to_sharegpt_single_token(r, img) for r in train], f, ensure_ascii=False)

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

    # 라벨 위치 분포(셔플 확인) + 통계
    from collections import Counter
    pos = Counter(r["label"] for r in train)
    n_amb_t = sum(r["label_type"] == "ambiguous" for r in train)
    n_amb_v = sum(r["label_type"] == "ambiguous" for r in val)
    print(f"[CLEAN] 고유문항 {len(allrecs)} (불량 {n_malformed}건 제외)")
    print(f"[CLEAN] VAL {len(val)} (amb {n_amb_v}/dis {len(val)-n_amb_v}) [시나리오 atomic]")
    print(f"[CLEAN] TRAIN {len(train)} (amb {n_amb_t}/dis {len(train)-n_amb_t}) [단일토큰]")
    print(f"[CLEAN] 정답 위치 분포(셔플 후, 균등해야 정상): 0={pos[0]} 1={pos[1]} 2={pos[2]}")
    print(f"[CLEAN] 시나리오 누출 = 0 ✅")


if __name__ == "__main__":
    main()
