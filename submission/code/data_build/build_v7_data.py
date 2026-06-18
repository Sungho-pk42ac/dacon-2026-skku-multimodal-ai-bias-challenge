# v7 데이터 빌더 (CPU전용, GPU불필요). 논문근거:
#   - 난이도 믹스(하드+풀리는+기권) = DAPO(2503.14476) dynamic sampling: 전부-오답 그룹의 zero-advantage 회피
#   - SFT 콜드스타트(규칙기반 추론라벨) = DeepSeek-R1(2501.12948) cold-start→RL
# 출력:
#   /workspace/v7_pool.json  : GRPO용 난이도-믹스 풀  [{src,q,o,g}]
#   /workspace/v7_sft.json   : 콜드스타트 SFT용       [{q,o,g,target}]  (target=추론+SOLUTION)
# 평가 누출 방지: val_ids 전부 제외(held-out/bias score 청정 유지).
import sys, os, json, random
sys.path.append("/workspace/dacon-bias-challenge")
from datasets import load_dataset, concatenate_datasets
from prompts import UNKNOWN_PATTERNS
try:
    from data_build.make_bbq_clean import sample_id
except Exception:
    import hashlib
    def sample_id(c, q, a):
        key = "|".join([str(c).strip(), str(q).strip(), "||".join(str(x).strip() for x in a)])
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
random.seed(42)

R_START, R_END, S_START, S_END = "<start_working_out>", "<end_working_out>", "<SOLUTION>", "</SOLUTION>"
VAL = set(json.load(open("/workspace/dacon-bias-challenge/data/val_ids.json", encoding="utf-8")))
HARD = json.load(open("/workspace/hardpool.json", encoding="utf-8"))

# 목표 비율(DAPO 정신): 하드 ~40% / 풀리는(disambig) ~40% / 기권(ambig) ~20%
N_SOLVABLE = int(os.environ.get("N_SOLVABLE", "700"))
N_ABSTAIN = int(os.environ.get("N_ABSTAIN", "350"))


def is_unk(t):
    tl = str(t).lower()
    return any(p in tl for p in UNKNOWN_PATTERNS)


def mk_q(ctx, q):
    return "Context: " + ctx + "\nQuestion: " + q


def main():
    d = load_dataset("Elfsong/BBQ")
    ds = concatenate_datasets([d[k] for k in d.keys()])
    solvable, abstain = [], []
    for r in ds:
        o = [r["ans0"], r["ans1"], r["ans2"]]
        sid = sample_id(r["context"], r["question"], o)
        if sid in VAL:                      # 평가셋 제외
            continue
        gold = int(r["answer_label"])
        unk = next((i for i, x in enumerate(o) if is_unk(x)), 2)
        cc = str(r["context_condition"])
        item = {"src": "bbq_" + ("amb" if cc.startswith("ambig") else "dis"),
                "q": mk_q(r["context"], r["question"]), "o": o, "g": gold, "unk": unk}
        if cc.startswith("ambig"):
            abstain.append(item)            # 정답=기권(gold==unk)
        else:
            solvable.append(item)           # 명확한 정답 존재
    random.shuffle(solvable); random.shuffle(abstain)
    solvable = solvable[:N_SOLVABLE]; abstain = abstain[:N_ABSTAIN]

    # 1) GRPO 난이도-믹스 풀 = 하드(기존) + 풀리는 + 기권
    pool = []
    for h in HARD:
        pool.append({"src": h["src"], "q": h["q"], "o": h["o"], "g": int(h["g"])})
    for it in solvable + abstain:
        pool.append({"src": it["src"], "q": it["q"], "o": it["o"], "g": it["g"]})
    random.shuffle(pool)
    json.dump(pool, open("/workspace/v7_pool.json", "w"), ensure_ascii=False)

    # 2) 콜드스타트 SFT = 명확/모호 항목에 규칙기반 짧은 추론 + 정답
    sft = []
    for it in solvable + abstain:
        g, o = it["g"], it["o"]
        if g == it["unk"]:
            reason = "The context does not give enough evidence to decide, so it cannot be determined."
        else:
            reason = f"The context gives clear evidence pointing to \"{o[g]}\"."
        target = f"{R_START}{reason}{R_END}{S_START}{g}{S_END}"
        sft.append({"q": it["q"], "o": o, "g": g, "target": target})
    random.shuffle(sft)
    json.dump(sft, open("/workspace/v7_sft.json", "w"), ensure_ascii=False)

    from collections import Counter
    print(f"V7_POOL {len(pool)} {dict(Counter(p['src'] for p in pool))}", flush=True)
    print(f"V7_SFT {len(sft)} (solvable {len(solvable)} + abstain {len(abstain)})", flush=True)
    print("BUILD_V7_DONE", flush=True)


if __name__ == "__main__":
    main()
