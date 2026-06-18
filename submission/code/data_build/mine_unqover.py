# UnQover 기권 하드네거티브 채굴(규칙-안전: 독립 공개 편향 진단셋, 대회 평가셋 비파생).
# underspecified 문항 → 정답=기권("Cannot be determined"). v4가 사람을 고르면(=성별 고정관념) 하드네거티브.
# 3지선다 [subjA, subjB, 기권] + 기권위치 셔플(position-bias 제거). 1.8GB → ijson 스트리밍+조기중단.
import sys, os, json, random, hashlib
sys.path.append("/workspace/dacon-bias-challenge")
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label, UNKNOWN_PATTERNS
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
random.seed(42)

SRC = "/workspace/unqover/data/gender"
HARDPOOL = "/workspace/hardpool.json"
CAND = int(os.environ.get("CAND", "3000"))   # 검사할 후보 문항 상한
WANT = int(os.environ.get("WANT", "400"))    # 채굴할 하드네거티브 상한
ABSTAIN = "Cannot be determined"
img = Image.open("/workspace/dacon-bias-challenge/data/placeholder.jpg").convert("RGB")


def make3(subjA, subjB, key):
    """[subjA, subjB, 기권] → 셔플(키 기반 결정적) → (options, 기권 인덱스)."""
    opts = [subjA, subjB, ABSTAIN]
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    order = [0, 1, 2]
    # 결정적 셔플(Fisher-Yates, 해시 시드)
    for i in range(2, 0, -1):
        j = h % (i + 1); h //= 7
        order[i], order[j] = order[j], order[i]
    shuf = [opts[i] for i in order]
    return shuf, shuf.index(ABSTAIN)


def stream_candidates():
    """ijson 없이 pretty-printed JSON을 brace-counting으로 라인 스트리밍(메모리 O(1)).
    top-level dict의 각 value 객체를 한 덩어리씩 json.loads → q0만 사용."""
    seen = set()
    with open(SRC, "r", encoding="utf-8") as f:
        f.readline()  # 최상위 '{' 스킵
        cur = []; depth = 0; started = False
        for line in f:
            s = line.rstrip("\n")
            if not started:
                # 최상위 키 라인:  '    "....": {'  → value 객체 시작
                st = s.strip()
                if st.startswith('"') and st.endswith("{"):
                    started = True; cur = ["{"]; depth = 1
                continue
            cur.append(s)
            depth += s.count("{") - s.count("}")
            if depth <= 0:
                started = False
                block = "\n".join(cur).rstrip().rstrip(",")
                try:
                    val = json.loads(block)
                    ctx = val["context"]; q0 = val["q0"]
                    a, b = q0["ans0"]["text"], q0["ans1"]["text"]
                    ques = q0["question"]
                except Exception:
                    cur = []; continue
                cur = []
                dk = (ctx, ques)
                if dk in seen:
                    continue
                seen.add(dk)
                opts, gold = make3(a, b, ctx + "|" + ques)
                yield {"context": ctx, "question": ques, "o": opts, "g": gold}
                if len(seen) >= CAND:
                    break


def main():
    p = "/workspace/dacon-bias-challenge/outputs/merged_v4"
    m = AutoModelForImageTextToText.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(p)

    def pred(q, o):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": USER_TEMPLATE.format(context="", question=q, options_block=format_options(o))}, {"type": "image", "image": img}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        im, vi = process_vision_info(msgs)
        inp = proc(text=[t], images=im, videos=vi, padding=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=4, do_sample=False)
        out = proc.batch_decode([x[len(i):] for i, x in zip(inp.input_ids, g)], skip_special_tokens=True)[0]
        return text_to_label(out, o)

    hard = []; n = 0; abstain_correct = 0
    for it in stream_candidates():
        n += 1
        q = "Context: " + it["context"] + "\nQuestion: " + it["question"]
        pr = pred(q, it["o"])
        if pr == it["g"]:
            abstain_correct += 1
        else:
            # v4가 기권 대신 사람을 고름 = 성별 고정관념 → 하드네거티브
            hard.append({"src": "unqover", "q": q, "o": it["o"], "g": it["g"]})
        if n % 200 == 0:
            print(f"[UNQ] checked {n} | hard {len(hard)} | v4기권정답 {abstain_correct}", flush=True)
        if len(hard) >= WANT:
            break
    print(f"[UNQ] FINAL checked {n} | hard {len(hard)} | v4 기권정답률 {abstain_correct/max(n,1):.3f}", flush=True)

    # 기존 하드풀에 합치기
    pool = json.load(open(HARDPOOL, encoding="utf-8"))
    before = len(pool)
    pool.extend(hard)
    random.shuffle(pool)
    json.dump(pool, open(HARDPOOL, "w"), ensure_ascii=False)
    print(f"UNQ_MERGED before={before} add={len(hard)} after={len(pool)}", flush=True)
    print("UNQ_DONE", flush=True)


if __name__ == "__main__":
    main()
