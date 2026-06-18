"""
v8B 집중 풀 빌더 (v8B Step 1).

목표: v8A가 유일하게 실패한 '옵션순서 강건성(셔플 일관성)'만 정밀 타격하되, OOD 이득은 보존.
방법: 공개데이터 후보를 v4·v8A 두 모델로 채점(원본+셔플) + v8A로 G회 동적샘플링 →
      6개 집중 카테고리 + 각 샘플의 셔플 짝(paired)을 풀에 포함.

집중 카테고리:
 1 v8A_shuffle_inconsistent : v8A가 옵션순서 바꾸면 의미적 답이 달라짐 (핵심 타깃)
 2 v8A_regression           : v4는 맞고 v8A는 틀림 (회귀 복구)
 3 both_wrong               : v4·v8A 둘 다 틀림 (하드)
 4 amb_person_error         : ambiguous인데 v8A가 unknown 대신 사람 선택
 5 dis_overabstain          : disambiguated인데 v8A가 과도하게 unknown 선택
 6 ood_v8a_gain             : OOD에서 v4는 틀리고 v8A는 맞음 (OOD 이득 보존용)

밸런스 규칙:
 - ambiguous-unknown 과표집 금지(상한), disambiguated 충분히 포함(과기권 방지), OOD 포함.
 - 가능한 모든 선택 샘플에 셔플 짝 추가(pair_id 공유, gold_text 동일) → 순서불변 직접 학습.
 - 소스 분포 로깅.

누출방지: DACON 평가셋 미사용. 공개데이터만. val_ids + v8_eval.json 항목은 풀에서 제외(평가셋과 분리).
출력: outputs/v8b_pool.json, outputs/v8b_pool_stats.json, docs/V8B_POOL_REPORT.md
       (편의상 /workspace/v8b_pool.json 사본도 저장 → 학습기 기본 경로)

실행: PER=260 G=8 python data_build/build_v8b_pool.py \
        --v4 outputs/merged_v4 --v8a outputs/merged_v8a --eval /workspace/v8_eval.json
"""
import argparse, json, os, sys, hashlib, random
from collections import Counter, defaultdict
sys.path.append("/workspace/dacon-bias-challenge")
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label, UNKNOWN_PATTERNS
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset, concatenate_datasets

random.seed(42)
R = "/workspace/dacon-bias-challenge"
PH = Image.open(R + "/data/placeholder.jpg").convert("RGB")
VAL = set(json.load(open(R + "/data/val_ids.json", encoding="utf-8")))


# ---- 데이터/스키마 헬퍼 (build_v8_robust_pool과 동일 로직; qwen_vl_utils 의존 제거 위해 인라인) ----
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
    idx = list(range(len(answers))); random.shuffle(idx)
    return [answers[i] for i in idx], idx.index(label)


def build_candidates():
    """공개데이터 → 통합스키마 후보 (val 제외). bbq_amb/bbq_dis/ood_*. (DACON 평가셋 미사용)"""
    cands = []
    CPER = int(os.environ.get("CPER", "900"))   # 소스당 원천 후보 상한(넉넉히 → 이후 PER로 재샘플)
    d = load_dataset("Elfsong/BBQ"); dsb = concatenate_datasets([d[k] for k in d.keys()])
    na = nd = 0
    for r in dsb:
        ans = [r["ans0"], r["ans1"], r["ans2"]]
        if sid(r["context"], r["question"], ans) in VAL:
            continue
        amb = str(r["context_condition"]).startswith("ambig")
        if amb and na >= CPER: continue
        if not amb and nd >= CPER: continue
        cands.append(rec("bbq_amb" if amb else "bbq_dis", r["context"], r["question"], ans,
                         int(r["answer_label"]),
                         meta={"polarity": r["question_polarity"], "target": int(r["target_label"])}))
        na += amb; nd += (not amb)
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
                if n >= CPER: break
        except Exception as e:
            print(f"[OOD] {src} fail {str(e)[:60]}", flush=True)
    return cands

# 카테고리별 상한(밸런스). ambiguous-unknown 과표집 금지가 핵심.
CAPS = {
    "v8a_shuffle_inconsistent": 320,   # 핵심 타깃 — 가장 많이
    "v8a_regression": 160,
    "both_wrong": 120,
    "amb_person_error": 90,            # amb 사람선택오류는 제한(과기권 유도 방지)
    "dis_overabstain": 120,            # dis 과기권은 충분히(과기권 억제 학습)
    "ood_v8a_gain": 120,               # OOD 이득 보존
}


def batched_predict(model, proc, triples, max_new=4, do_sample=False, temperature=0.9, batch=32):
    """triples=[(ctx,q,answers)] → [예측 인덱스]. 결정/샘플 공용."""
    preds = []
    for i in range(0, len(triples), batch):
        ch = triples[i:i + batch]
        texts = []
        for ctx, q, ans in ch:
            user = USER_TEMPLATE.format(context=ctx, question=q, options_block=format_options(ans))
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": PH}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = proc(text=texts, images=[PH] * len(ch), padding=True, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=max_new, do_sample=do_sample)
        if do_sample:
            kw.update(temperature=temperature, top_p=0.95)
        with torch.no_grad():
            g = model.generate(**inp, **kw)
        outs = proc.batch_decode([r[inp.input_ids.shape[1]:] for r in g], skip_special_tokens=True)
        preds.extend(text_to_label(outs[j], ch[j][2]) for j in range(len(ch)))
    return preds


def load_model(path):
    proc = AutoProcessor.from_pretrained(path)
    proc.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    return m, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default="outputs/merged_v4")
    ap.add_argument("--v8a", default="outputs/merged_v8a")
    ap.add_argument("--eval", default="/workspace/v8_eval.json")
    ap.add_argument("--G", type=int, default=int(os.environ.get("G", "8")))
    args = ap.parse_args()

    # 후보 생성(공개데이터) + 평가셋/val 분리
    cands = build_candidates()
    eval_sids = set()
    if os.path.exists(args.eval):
        for r in json.load(open(args.eval, encoding="utf-8")):
            eval_sids.add(sid(r.get("context", ""), r["question"], r["answers"]))
    cands = [c for c in cands if sid(c["context"], c["question"], c["answers"]) not in eval_sids]
    # 후보가 너무 많으면 소스별 샘플링(런타임 관리). 셔플불일치는 많이 봐야 하므로 넉넉히.
    PER = int(os.environ.get("PER", "260"))
    bysrc = defaultdict(list)
    for c in cands:
        bysrc[c["src"]].append(c)
    cands = []
    for s, items in bysrc.items():
        random.shuffle(items)
        cands.extend(items[:PER])
    print(f"[CAND] {len(cands)} {dict(Counter(c['src'] for c in cands))}", flush=True)

    triples = [(c.get("context", ""), c["question"], c["answers"]) for c in cands]
    # 셔플 변형(채점·짝 생성 공용): 각 후보의 셔플 답/라벨을 한 번만 결정(재현성)
    shufs = [shuffle_perm(c["answers"], c["label"]) for c in cands]
    triples_shuf = [(cands[i].get("context", ""), cands[i]["question"], shufs[i][0]) for i in range(len(cands))]

    # --- v4 채점 ---
    print("[LOAD] v4", flush=True)
    m4, p4 = load_model(args.v4)
    v4_orig = batched_predict(m4, p4, triples)
    v4_shuf = batched_predict(m4, p4, triples_shuf)
    del m4
    torch.cuda.empty_cache()

    # --- v8a 채점 + 동적 G샘플링 ---
    print("[LOAD] v8a", flush=True)
    m8, p8 = load_model(args.v8a)
    v8_orig = batched_predict(m8, p8, triples)
    v8_shuf = batched_predict(m8, p8, triples_shuf)
    # 동적 k: 각 후보 G회 샘플링 → 정답수
    dyn_k = [0] * len(cands)
    for g in range(args.G):
        gp = batched_predict(m8, p8, triples, do_sample=True, temperature=0.9)
        for i, c in enumerate(cands):
            dyn_k[i] += int(gp[i] == int(c["label"]))
        print(f"[DYN] round {g+1}/{args.G}", flush=True)
    del m8
    torch.cuda.empty_cache()

    # --- 태깅 ---
    def sem(ans, idx):
        return ans[idx] if 0 <= idx < len(ans) else None

    src_correct = defaultdict(int); src_total = defaultdict(int)
    tagged = []
    easy_cnt = hard_cnt = useful_cnt = 0
    for i, c in enumerate(cands):
        lab = int(c["label"]); u = c["unknown_idx"]; src = c["src"]
        src_total[src] += 1; src_correct[src] += int(v8_orig[i] == lab)
        k = dyn_k[i]
        if k == 0: hard_cnt += 1
        elif k == args.G: easy_cnt += 1
        else: useful_cnt += 1
        tags = []
        # 1 셔플 불일치(v8a): 원본 예측의 '의미'와 셔플 예측의 '의미'가 다른가
        if sem(c["answers"], v8_orig[i]) != sem(shufs[i][0], v8_shuf[i]):
            tags.append("v8a_shuffle_inconsistent")
        # 2 회귀: v4 정답 & v8a 오답
        if v4_orig[i] == lab and v8_orig[i] != lab:
            tags.append("v8a_regression")
        # 3 양쪽 오답
        if v4_orig[i] != lab and v8_orig[i] != lab:
            tags.append("both_wrong")
        # 4 amb 사람선택오류
        if src == "bbq_amb" and u is not None and v8_orig[i] != u:
            tags.append("amb_person_error")
        # 5 dis 과기권
        if src == "bbq_dis" and u is not None and v8_orig[i] == u and lab != u:
            tags.append("dis_overabstain")
        # 6 OOD 이득 보존
        if src.startswith("ood") and v4_orig[i] != lab and v8_orig[i] == lab:
            tags.append("ood_v8a_gain")
        if tags:
            c["meta"]["tags"] = tags
            c["meta"]["dyn_k"] = k; c["meta"]["dyn_G"] = args.G
            c["meta"]["v4_orig"] = int(v4_orig[i]); c["meta"]["v8a_orig"] = int(v8_orig[i])
            c["meta"]["shuffle"] = {"answers": shufs[i][0], "label": int(shufs[i][1])}
            tagged.append(c)

    src_base = {s: round(src_correct[s] / max(1, src_total[s]), 4) for s in src_total}

    # --- 카테고리 상한 적용해 선택(밸런스) ---
    random.shuffle(tagged)
    picked, per_cat = [], Counter()
    seen_sid = set()
    # 우선순위: 셔플불일치 > 회귀 > dis과기권 > OOD이득 > amb오류 > 양쪽오답
    prio = ["v8a_shuffle_inconsistent", "v8a_regression", "dis_overabstain",
            "ood_v8a_gain", "amb_person_error", "both_wrong"]
    for cat in prio:
        for c in tagged:
            s = sid(c["context"], c["question"], c["answers"])
            if s in seen_sid:
                continue
            if cat in c["meta"]["tags"] and per_cat[cat] < CAPS[cat]:
                picked.append(c); seen_sid.add(s)
                for t in c["meta"]["tags"]:
                    per_cat[t] += 1

    # --- 셔플 짝(paired) 추가: 각 선택 샘플의 원본 + 셔플변형을 별도 행으로 ---
    pool = []
    for c in picked:
        gold_text = c["answers"][int(c["label"])]
        pid = sid(c["context"], c["question"], c["answers"])
        base = dict(c); base["gold_text"] = gold_text; base["pair_id"] = pid; base["variant"] = "orig"
        pool.append(base)
        sa, sl = c["meta"]["shuffle"]["answers"], c["meta"]["shuffle"]["label"]
        twin = dict(c)
        twin["answers"] = sa; twin["label"] = int(sl)
        twin["unknown_idx"] = unknown_idx(sa)
        twin["gold_text"] = gold_text; twin["pair_id"] = pid; twin["variant"] = "shuf"
        twin["meta"] = dict(c["meta"]); twin["meta"]["tags"] = c["meta"]["tags"] + ["paired_shuffle"]
        pool.append(twin)

    random.shuffle(pool)
    os.makedirs(R + "/outputs", exist_ok=True)
    os.makedirs(R + "/docs", exist_ok=True)
    json.dump(pool, open(R + "/outputs/v8b_pool.json", "w"), ensure_ascii=False)
    json.dump(pool, open("/workspace/v8b_pool.json", "w"), ensure_ascii=False)  # 학습기 기본경로 사본

    stats = {
        "candidates_scored": len(cands),
        "tagged_total": len(tagged),
        "picked_unique": len(picked),
        "pool_total_with_pairs": len(pool),
        "per_category": dict(per_cat),
        "pool_by_src": dict(Counter(c["src"] for c in pool)),
        "pool_by_variant": dict(Counter(c["variant"] for c in pool)),
        "src_base_acc_v8a": src_base,          # source_normalized reward 기준선
        "dynamic": {"easy_count": easy_cnt, "hard_count": hard_cnt,
                    "useful_count": useful_cnt, "G": args.G},
        "caps": CAPS, "per_source_cap": PER,
    }
    json.dump(stats, open(R + "/outputs/v8b_pool_stats.json", "w"), ensure_ascii=False, indent=2)
    json.dump(stats, open("/workspace/v8b_pool_stats.json", "w"), ensure_ascii=False, indent=2)

    # --- docs/V8B_POOL_REPORT.md ---
    with open(R + "/docs/V8B_POOL_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# v8B 집중 풀 리포트\n\n")
        f.write("> 목표: v8A가 실패한 옵션순서 강건성만 정밀 타격(OOD 이득 보존). 공개데이터만, DACON 평가셋 미사용.\n")
        f.write("> 평가셋(v8_eval.json)·val_ids 항목은 풀에서 제외 → 학습/평가 분리.\n\n")
        f.write(f"- 채점 후보: **{len(cands)}** (소스당 상한 {PER})\n")
        f.write(f"- 태깅된 샘플: **{len(tagged)}** / 선택(고유): **{len(picked)}** / 셔플짝 포함 풀: **{len(pool)}**\n\n")
        f.write("## 카테고리 분포 (선택 후, 상한 적용)\n\n| 카테고리 | 개수 | 상한 |\n|---|---:|---:|\n")
        for cat in prio + ["paired_shuffle"]:
            f.write(f"| {cat} | {per_cat.get(cat, 0)} | {CAPS.get(cat, '-')} |\n")
        f.write("\n## 소스 분포\n\n| 소스 | 풀 개수 | v8A 기준정확도 |\n|---|---:|---:|\n")
        for s in sorted(stats["pool_by_src"]):
            f.write(f"| {s} | {stats['pool_by_src'][s]} | {src_base.get(s, '-')} |\n")
        f.write(f"\n## 동적 샘플링(v8A, G={args.G})\n\n")
        f.write(f"- useful(0<k<G): **{useful_cnt}** · easy(k=G): {easy_cnt} · hard(k=0): {hard_cnt}\n")
        f.write("- 풀은 useful 위주 카테고리로 구성(셔플불일치·회귀·과기권 등은 본질적으로 불확실 샘플).\n\n")
        f.write("## variant 분포(셔플 짝)\n\n")
        for v, n in stats["pool_by_variant"].items():
            f.write(f"- {v}: {n}\n")
        f.write("\n각 선택 샘플은 원본 + 셔플변형 2행으로 포함 → 순서불변(option-shuffle consistency) 직접 학습.\n")

    print("V8B_POOL_STATS", json.dumps(stats, ensure_ascii=False), flush=True)
    print("BUILD_V8B_DONE", flush=True)


if __name__ == "__main__":
    main()
