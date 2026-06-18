# v8 robust pool 빌더 (Phase 2+3). 통합 스키마(context/question 분리) + v4 기반 강건성 채굴.
# 누출방지: DACON 평가셋 미사용. 공개데이터(Elfsong/BBQ, siqa/csqa/obqa/arc)만. val_ids 학습제외.
# 채굴 카테고리: ①v4오답 ②셔플불안정(옵션순서 바꾸면 예측 바뀜) ③ambiguous 사람선택오류
#               ④disambiguated 과기권 ⑤OOD 하드 + 난이도믹스용 일부 정답샘플(DAPO).
# 출력: /workspace/v8_pool.json, /workspace/v8_eval.json, /workspace/v8_pool_stats.json
import sys, os, json, random, hashlib
sys.path.append("/workspace/dacon-bias-challenge")
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label, UNKNOWN_PATTERNS
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from datasets import load_dataset, concatenate_datasets
random.seed(42)

V4 = "/workspace/dacon-bias-challenge/outputs/merged_v4"
VAL = set(json.load(open("/workspace/dacon-bias-challenge/data/val_ids.json", encoding="utf-8")))
IMG = Image.open("/workspace/dacon-bias-challenge/data/placeholder.jpg").convert("RGB")
PER = int(os.environ.get("PER", "500"))   # 소스당 후보 상한
EVAL_PER = int(os.environ.get("EVAL_PER", "200"))


def sid(c, q, a):
    key = "|".join([str(c).strip(), str(q).strip(), "||".join(str(x).strip() for x in a)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def is_unk(t):
    tl = str(t).lower()
    return any(p in tl for p in UNKNOWN_PATTERNS)


def unknown_idx(answers):
    for i, a in enumerate(answers):
        if is_unk(a):
            return i
    return None


def rec(src, context, question, answers, label, image_path=None, meta=None):
    """통합 스키마 레코드."""
    return {"src": src, "context": context, "question": question, "answers": answers,
            "label": int(label), "unknown_idx": unknown_idx(answers),
            "image_path": image_path, "meta": meta or {}}


def reduce3(opts, gold):
    if len(opts) <= 3:
        return opts, gold
    others = [i for i in range(len(opts)) if i != gold]
    keep = [gold] + random.sample(others, 2); random.shuffle(keep)
    return [opts[i] for i in keep], keep.index(gold)


def shuffle_perm(answers, label):
    """옵션순서 셔플 + 새 label. (셔플 일관성 검사용)"""
    idx = list(range(len(answers)))
    random.shuffle(idx)
    return [answers[i] for i in idx], idx.index(label)


def build_candidates():
    """공개데이터 → 통합스키마 후보 (val 제외). bbq_amb/bbq_dis/ood_*."""
    cands = []
    # BBQ (메타데이터: context_condition, answer_label)
    d = load_dataset("Elfsong/BBQ"); ds = concatenate_datasets([d[k] for k in d.keys()])
    seen = set(); na = nd = 0
    for r in ds:
        ans = [r["ans0"], r["ans1"], r["ans2"]]
        s = sid(r["context"], r["question"], ans)
        if s in VAL:
            continue
        amb = str(r["context_condition"]).startswith("ambig")
        src = "bbq_amb" if amb else "bbq_dis"
        if amb and na >= PER + EVAL_PER: continue
        if not amb and nd >= PER + EVAL_PER: continue
        cands.append(rec(src, r["context"], r["question"], ans, int(r["answer_label"]),
                         meta={"polarity": r["question_polarity"], "target": int(r["target_label"])}))
        na += amb; nd += (not amb)
    # OOD 일반추론 (train split → val_ids와 무관, 3지선다 축소)
    ood = [("ood_siqa", lambda: load_dataset("lighteval/siqa", split="train"), "siqa"),
           ("ood_csqa", lambda: load_dataset("tau/commonsense_qa", split="train"), "mc"),
           ("ood_obqa", lambda: load_dataset("allenai/openbookqa", "main", split="train"), "mc"),
           ("ood_arc", lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="train"), "mc")]
    for src, loader, kind in ood:
        try:
            dd = loader(); n = 0
            for r in dd:
                if kind == "siqa":
                    g = int(r["label"]) - 1
                    if g not in (0, 1, 2): continue
                    ctx, q, o = r["context"], r["question"], [r["answerA"], r["answerB"], r["answerC"]]
                else:
                    txt, labs = r["choices"]["text"], r["choices"]["label"]
                    if r["answerKey"] not in labs: continue
                    g0 = labs.index(r["answerKey"]); o, g = reduce3(txt, g0)
                    ctx, q = "", (r["question"] if "question" in r else r["question_stem"])
                    cands.append(rec(src, ctx, q, o, g)); n += 1
                    if n >= PER + EVAL_PER: break
                    continue
                cands.append(rec(src, ctx, q, o, g)); n += 1
                if n >= PER + EVAL_PER: break
        except Exception as e:
            print(f"[OOD] {src} fail {str(e)[:60]}", flush=True)
    return cands


def main():
    cands = build_candidates()
    from collections import Counter
    print(f"[CAND] {len(cands)} {dict(Counter(c['src'] for c in cands))}", flush=True)

    m = AutoModelForImageTextToText.from_pretrained(V4, torch_dtype=torch.bfloat16, device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(V4)

    def pred(context, question, answers):
        user = USER_TEMPLATE.format(context=context, question=question, options_block=format_options(answers))
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": IMG}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        im, vi = process_vision_info(msgs)
        inp = proc(text=[t], images=im, videos=vi, padding=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=4, do_sample=False)
        out = proc.batch_decode([x[len(i):] for i, x in zip(inp.input_ids, g)], skip_special_tokens=True)[0]
        return text_to_label(out, answers)

    # 소스별 train/eval 분리 후 v4로 강건성 분류
    bycat = {}
    for c in cands:
        bycat.setdefault(c["src"], []).append(c)
    pool, evalset = [], []
    for src, items in bycat.items():
        random.shuffle(items)
        ev, tr = items[:EVAL_PER], items[EVAL_PER:PER + EVAL_PER]
        evalset.extend(ev)
        for c in tr:
            p0 = pred(c["context"], c["question"], c["answers"])
            sa, sl = shuffle_perm(c["answers"], c["label"])
            p1 = pred(c["context"], c["question"], sa)
            # 의미적 일치: 원래 예측이 가리킨 답이 셔플에서도 같은 답인가
            sem0 = c["answers"][p0] if 0 <= p0 < 3 else None
            sem1 = sa[p1] if 0 <= p1 < 3 else None
            tags = []
            if p0 != c["label"]:
                tags.append("v4_wrong")
            if sem0 != sem1:
                tags.append("unstable")
            if c["unknown_idx"] is not None and src == "bbq_amb" and p0 != c["unknown_idx"]:
                tags.append("amb_person_error")
            if c["unknown_idx"] is not None and src == "bbq_dis" and p0 == c["unknown_idx"] and c["label"] != c["unknown_idx"]:
                tags.append("dis_overabstain")
            if tags:
                c["meta"]["tags"] = tags
                c["meta"]["shuffle"] = {"answers": sa, "label": sl}  # 셔플쌍(보상용)
                pool.append(c)
        # 난이도믹스(DAPO): 각 소스에서 v4가 맞춘 쉬운샘플 일부도 포함 → reward variance 확보
        easy = [c for c in tr if c not in pool][:max(1, len(tr) // 5)]
        for c in easy:
            c["meta"]["tags"] = ["easy_anchor"]
        pool.extend(easy)

    random.shuffle(pool)
    json.dump(pool, open("/workspace/v8_pool.json", "w"), ensure_ascii=False)
    json.dump(evalset, open("/workspace/v8_eval.json", "w"), ensure_ascii=False)
    tagc = Counter(t for c in pool for t in c["meta"].get("tags", []))
    stats = {"pool_total": len(pool), "eval_total": len(evalset),
             "pool_by_src": dict(Counter(c["src"] for c in pool)),
             "eval_by_src": dict(Counter(c["src"] for c in evalset)),
             "pool_by_tag": dict(tagc)}
    json.dump(stats, open("/workspace/v8_pool_stats.json", "w"), ensure_ascii=False, indent=2)
    print("V8_POOL_STATS", json.dumps(stats, ensure_ascii=False), flush=True)
    print("BUILD_V8_DONE", flush=True)


if __name__ == "__main__":
    main()
