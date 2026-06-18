"""
일반화 검수용 '완전 미관측' held-out eval 빌더 (OOD-v2).
어떤 버전(v1~v8b)도 학습에 사용하지 않은 공개 MC 데이터셋으로 순수 일반화 측정.
포함: MMLU(지식), HellaSwag(상식), ARC-Challenge(어려운 추론), PIQA(물리상식), WinoGrande(상식).
각 소스당 EVAL_PER(기본 150), 4지선다는 reduce3로 3지선다 축소, 2지선다는 그대로.
출력: /workspace/ood2_eval.json (통합스키마; unknown_idx=None → 순수 정확도 채점).
누출방지: 모두 validation/test split 사용(가능 시), DACON 무관 공개데이터.

실행: EVAL_PER=150 python3 data_build/build_ood2.py
"""
import sys, os, json, random
sys.path.append("/workspace/dacon-bias-challenge")
sys.path.append("/workspace/dacon-bias-challenge/data_build")
from build_v8b_pool import rec, reduce3   # 통합스키마 rec(), 4->3 축소
from datasets import load_dataset
from collections import Counter
random.seed(42)
EVAL_PER = int(os.environ.get("EVAL_PER", "150"))


def take(src, gen, n=EVAL_PER):
    out = []
    try:
        for r in gen():
            rr = r()
            if rr is None:
                continue
            ctx, q, opts, g = rr
            if not opts or g is None or not (0 <= g < len(opts)):
                continue
            if len(opts) > 3:
                opts, g = reduce3(opts, g)
            out.append(rec(src, ctx, q, opts, g))
            if len(out) >= n:
                break
    except Exception as e:
        print(f"[{src}] fail {str(e)[:80]}", flush=True)
    print(f"[{src}] {len(out)}", flush=True)
    return out


def main():
    evalset = []

    # MMLU (4지선다 지식) — validation split
    def mmlu():
        d = load_dataset("cais/mmlu", "all", split="validation")
        for r in d:
            yield (lambda r=r: ("", r["question"], list(r["choices"]), int(r["answer"])))
    evalset += take("ood2_mmlu", mmlu)

    # HellaSwag (4지선다 상식) — validation
    def hella():
        d = load_dataset("Rowan/hellaswag", split="validation")
        for r in d:
            if r["label"] == "":
                continue
            yield (lambda r=r: (r["ctx"], "What happens next?", list(r["endings"]), int(r["label"])))
    evalset += take("ood2_hellaswag", hella)

    # ARC-Challenge (우리가 쓴 Easy와 다른 어려운 셋) — test
    def arcc():
        d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for r in d:
            txt, labs = r["choices"]["text"], r["choices"]["label"]
            if r["answerKey"] not in labs:
                continue
            g = labs.index(r["answerKey"])
            yield (lambda r=r, txt=txt, g=g: ("", r["question"], list(txt), g))
    evalset += take("ood2_arcc", arcc)

    # PIQA (2지선다 물리상식) — validation
    def piqa():
        d = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        for r in d:
            yield (lambda r=r: ("", r["goal"], [r["sol1"], r["sol2"]], int(r["label"])))
    evalset += take("ood2_piqa", piqa)

    # WinoGrande (2지선다 상식 코어퍼런스) — validation
    def wino():
        d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        for r in d:
            if r["answer"] not in ("1", "2"):
                continue
            yield (lambda r=r: (r["sentence"], "Which option fills the blank?",
                                [r["option1"], r["option2"]], int(r["answer"]) - 1))
    evalset += take("ood2_wino", wino)

    random.shuffle(evalset)
    json.dump(evalset, open("/workspace/ood2_eval.json", "w"), ensure_ascii=False)
    print("OOD2_BUILD_DONE", len(evalset), dict(Counter(c["src"] for c in evalset)), flush=True)


if __name__ == "__main__":
    main()
