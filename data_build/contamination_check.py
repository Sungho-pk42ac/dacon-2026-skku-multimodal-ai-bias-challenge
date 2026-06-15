"""
누출 검사 (Tier 1) — 우리 학습/검증 데이터가 대회 test.csv와 겹치지 않는지 검사.

대회 test는 BBQ 파생이므로, 우리 BBQ 학습데이터에 test와 동일/유사 문항이 섞이면 = Data Leakage.
운영진 공지: "Public=오픈벤치 샘플". 즉 test의 Public 부분이 BBQ면 우리 BBQ 학습이 그걸 외울 수 있음.
→ context+question 기준으로 (1)정규화 완전일치 (2)8-gram Jaccard 중복을 검출·제거.

입력: data/bbq_v4_train.json, data/bbq_val_clean.jsonl, data_build/test_full.csv
출력: contamination_report.json, removed_samples.jsonl
실행: python data_build/contamination_check.py --threshold 0.6
"""

import argparse
import csv
import json
import os
import re
import sys


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def ngrams(text, n=8):
    toks = norm(text).split()
    return set(tuple(toks[i:i + n]) for i in range(max(len(toks) - n + 1, 0))) or {tuple(toks)}


def load_train(path):
    """bbq_v4_train.json(sharegpt) → context+question 텍스트 추출."""
    out = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for i, rec in enumerate(data):
        user = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
        out.append({"id": f"train_{i}", "text": user})
    return out


def load_val(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            out.append({"id": r.get("sample_id", f"val_{i}"), "text": r["context"] + " " + r["question"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data_build/test_full.csv")
    ap.add_argument("--train", default="data/bbq_v4_train.json")
    ap.add_argument("--val", default="data/bbq_val_clean.jsonl")
    ap.add_argument("--n", type=int, default=8, help="n-gram 크기")
    ap.add_argument("--threshold", type=float, default=0.6, help="Jaccard 중복 임계")
    ap.add_argument("--out", default="contamination_report.json")
    args = ap.parse_args()

    # test: context+question 정규화 + n-gram 인덱스
    test_norm = set()
    test_ngrams = set()
    with open(args.test, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = r["context"] + " " + r["question"]
            test_norm.add(norm(t))
            test_ngrams |= ngrams(t, args.n)
    print(f"[test] {len(test_norm)} 문항, {len(test_ngrams)} 8-grams")

    removed = []
    stats = {"exact": 0, "high_overlap": 0, "checked": 0}
    for split, items in [("train", load_train(args.train)), ("val", load_val(args.val))]:
        for it in items:
            stats["checked"] += 1
            nt = norm(it["text"])
            if nt in test_norm:                      # (1) 정규화 완전일치
                removed.append({**it, "split": split, "reason": "exact"})
                stats["exact"] += 1
                continue
            g = ngrams(it["text"], args.n)           # (2) 8-gram Jaccard
            inter = len(g & test_ngrams)
            jac = inter / max(len(g), 1)
            if jac >= args.threshold:
                removed.append({**it, "split": split, "reason": f"overlap_{jac:.2f}"})
                stats["high_overlap"] += 1

    report = {
        "test_items": len(test_norm),
        "checked": stats["checked"],
        "exact_match": stats["exact"],
        "high_overlap": stats["high_overlap"],
        "contaminated_total": len(removed),
        "contamination_rate": round(len(removed) / max(stats["checked"], 1), 5),
        "threshold": args.threshold, "ngram": args.n,
        "verdict": "CLEAN ✅" if len(removed) == 0 else f"⚠️ {len(removed)}건 중복 — 제거 권고",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("removed_samples.jsonl", "w", encoding="utf-8") as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"CONTAM_RESULT exact={stats['exact']} overlap={stats['high_overlap']} rate={report['contamination_rate']}")


if __name__ == "__main__":
    main()
