"""
v6 하드네거티브 풀 채굴기 (mine_hard.py) — grpo_hard_v6.py의 입력 `hardpool.json`을 생성.

grpo_hard_v6.py가 기대하는 형식: 각 레코드 {"src": str, "q": str, "o": [3 options], "g": int}.
(grpo_hard_v6는 context=""로 고정하고 it["q"]를 question에 넣으므로, BBQ 앵커는 context를 q에 접합한다.)

채굴 로직(누출 안전 — 모두 독립 공개 QA, DACON 평가셋 비파생):
  1) 공개 일반추론 QA 로드: SIQA / CommonsenseQA / OpenBookQA / ARC-Easy(train split). 4지선다는 reduce3로 3지선다 축소.
  2) val_ids(SSOT, make_bbq_clean.py 생성) + v8_eval/ood2 누출 항목 sha1 sid로 전역배제.
  3) merged_v4로 greedy 추론(placeholder 이미지) → **v4가 틀린(오답) 문항만** 채택 = 하드네거티브(reward 신호 유지).
  4) BBQ-ambiguous 앵커 추가(기권정책 망각방지): ambiguous 문항(정답=기권)을 일부 포함.
  5) (선택) UnQover 기권 하드네거티브: data_build/mine_unqover.py 산출물이 있으면 합류.
  6) shuffle(seed 42) 후 hardpool.json 저장.

본 스크립트는 build_v8_robust_pool.py의 검증된 build_candidates()/pred() 로직과 동일한 데이터·스코어링을 사용하며,
v6용 단순 형식만 출력한다(따라서 v6 하드풀을 결정론적으로 재생성).

실행: python data_build/mine_hard.py --base outputs/merged_v4 --out /workspace/hardpool.json --per 500
"""
import argparse, json, os, sys, random, hashlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label, UNKNOWN_PATTERNS
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset

SEED = 42
random.seed(SEED)


def sid(c, q, a):
    """누출배제용 결정론적 sample_id (make_bbq_clean / build_v8_robust_pool와 동일 규칙)."""
    key = "|".join([str(c).strip(), str(q).strip(), "||".join(str(x).strip() for x in a)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def is_unk(t):
    tl = str(t).lower()
    return any(p in tl for p in UNKNOWN_PATTERNS)


def reduce3(opts, gold, rnd):
    """4지선다→3지선다(정답 보존)."""
    if len(opts) <= 3:
        return opts, gold
    others = [i for i in range(len(opts)) if i != gold]
    keep = [gold] + rnd.sample(others, 2)
    rnd.shuffle(keep)
    return [opts[i] for i in keep], keep.index(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--out", default="/workspace/hardpool.json")
    ap.add_argument("--val_ids", default="data/val_ids.json")
    ap.add_argument("--placeholder", default="data/placeholder.jpg")
    ap.add_argument("--per", type=int, default=500, help="OOD 소스당 후보 상한")
    ap.add_argument("--bbq_anchors", type=int, default=300, help="BBQ-ambiguous 앵커 수")
    ap.add_argument("--unqover", default="", help="(선택) mine_unqover.py 산출물 경로")
    args = ap.parse_args()
    rnd = random.Random(SEED)

    VAL = set(json.load(open(args.val_ids, encoding="utf-8"))) if os.path.exists(args.val_ids) else set()
    print(f"[mine_hard] val_ids {len(VAL)} 제외", flush=True)
    if os.path.exists(args.placeholder):
        IMG = Image.open(args.placeholder).convert("RGB")
    else:
        IMG = Image.new("RGB", (336, 336), (127, 127, 127))

    # ── 1) 공개 일반추론 QA 후보 (context="" + question=q, 3지선다) ──
    cands = []   # {"src","q","o","g"}
    ood = [("siqa", lambda: load_dataset("lighteval/siqa", split="train"), "siqa"),
           ("csqa", lambda: load_dataset("tau/commonsense_qa", split="train"), "mc"),
           ("obqa", lambda: load_dataset("allenai/openbookqa", "main", split="train"), "mc"),
           ("arc",  lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="train"), "mc")]
    for src, loader, kind in ood:
        try:
            dd = loader()
            n = 0   # 이 소스에서 채택한 후보 수
            for r in dd:
                if kind == "siqa":
                    g = int(r["label"]) - 1
                    if g not in (0, 1, 2):
                        continue
                    q, o = r["question"], [r["answerA"], r["answerB"], r["answerC"]]
                    ctx_for_sid = r["context"]
                    q = f"{r['context']} {r['question']}".strip()
                else:
                    txt, labs = r["choices"]["text"], r["choices"]["label"]
                    if r["answerKey"] not in labs:
                        continue
                    g0 = labs.index(r["answerKey"])
                    o, g = reduce3(list(txt), g0, rnd)   # 4지선다 → 3지선다(정답 보존)
                    q = r["question"] if "question" in r else r["question_stem"]
                    ctx_for_sid = ""
                if len(o) != 3:
                    continue
                if sid(ctx_for_sid, q, o) in VAL:   # 누출배제
                    continue
                cands.append({"src": "ood_" + src, "q": q, "o": list(o), "g": int(g)})
                n += 1
                if n >= args.per:
                    break
            print(f"[mine_hard] {src}: {n} 후보", flush=True)
        except Exception as e:
            print(f"[mine_hard] {src} 로드 실패 {str(e)[:80]}", flush=True)

    # ── 2) 모델 로드 + greedy 채점 → v4 오답만(하드네거티브) ──
    proc = AutoProcessor.from_pretrained(args.base)
    proc.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()

    def pred(q, o):
        user = USER_TEMPLATE.format(context="", question=q, options_block=format_options(o))
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": IMG}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t], images=[IMG], padding=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=4, do_sample=False)
        out = proc.batch_decode([x[inp.input_ids.shape[1]:] for x in g], skip_special_tokens=True)[0]
        return text_to_label(out, o)

    hard = []
    for c in cands:
        if pred(c["q"], c["o"]) != c["g"]:   # v4가 틀림 = 하드네거티브
            hard.append(c)
    print(f"[mine_hard] v4-오답 하드네거티브 {len(hard)}/{len(cands)}", flush=True)

    # ── 3) BBQ-ambiguous 앵커(기권정책 보존) ──
    try:
        from datasets import concatenate_datasets
        d = load_dataset("Elfsong/BBQ")
        ds = concatenate_datasets([d[k] for k in d.keys()])
        rows = []
        for r in ds:
            if not str(r["context_condition"]).startswith("ambig"):
                continue
            ans = [r["ans0"], r["ans1"], r["ans2"]]
            if sid(r["context"], r["question"], ans) in VAL:
                continue
            rows.append({"src": "bbq_amb_anchor",
                         "q": f"{r['context']} {r['question']}".strip(),
                         "o": ans, "g": int(r["answer_label"])})
        rnd.shuffle(rows)
        anchors = rows[:args.bbq_anchors]
        hard.extend(anchors)
        print(f"[mine_hard] BBQ-amb 앵커 {len(anchors)}", flush=True)
    except Exception as e:
        print(f"[mine_hard] BBQ 앵커 실패 {str(e)[:80]}", flush=True)

    # ── 4) (선택) UnQover 기권 하드네거티브 ──
    if args.unqover and os.path.exists(args.unqover):
        try:
            uq = json.load(open(args.unqover, encoding="utf-8"))
            uq = [r for r in uq if isinstance(r.get("o"), list) and len(r["o"]) == 3]
            hard.extend(uq)
            print(f"[mine_hard] UnQover {len(uq)} 합류", flush=True)
        except Exception as e:
            print(f"[mine_hard] UnQover 실패 {str(e)[:80]}", flush=True)

    # ── 5) 저장 ──
    rnd.shuffle(hard)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(hard, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    from collections import Counter
    print("MINE_HARD_DONE", args.out, len(hard), dict(Counter(c["src"] for c in hard)), flush=True)


if __name__ == "__main__":
    main()
