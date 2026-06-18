"""
v2 학습데이터: BBQ 각 문항에 **진짜 추론(CoT)**을 GPT로 생성 → 추론형 sharegpt.

v1의 한계(2개 고정문장 = 가짜 추론, loss 안 흔들림)를 해결.
받으신 prepare_dataset_0412.ipynb 철학과 동일: 정답을 아는 상태에서 그 답이 왜 맞는지 추론을 붙임.

data_build 단계 (OPENAI_API_KEY 사용, 추론 파이프라인과 분리). GPU 불필요.
출력: <out>/bbq_reason_train.json (LLaMA-Factory sharegpt) + <out>/dataset_info.json(키 bbq_reasoning_real)
resume 지원: 중간 캐시 jsonl에 이어쓰기.

실행: OPENAI_API_KEY=... python data_build/make_bbq_reasoning.py --n 5000 --model gpt-4o-mini --workers 8
"""

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, UNKNOWN_PATTERNS

SEED = 42
random.seed(SEED)
_lock = threading.Lock()


def is_unknown(opt):
    o = str(opt).lower()
    return any(p in o for p in UNKNOWN_PATTERNS)


def make_placeholder(path):
    from PIL import Image
    if not os.path.exists(path):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(path)


def gen_reason(client, model, ctx, q, answers, label, label_type):
    """정답(label)을 아는 상태에서 그 답에 대한 1~2문장 추론 생성."""
    gold = answers[label]
    if label_type == "ambiguous":
        ask = (f"Context: {ctx}\nQuestion: {q}\nOptions: {answers}\n"
               f"The correct answer is '{gold}' (option {label}), because the context does NOT give "
               f"enough evidence to pick a specific person and guessing would rely on a stereotype. "
               f"Write a concise 1-2 sentence reasoning that explains this WITHOUT using stereotypes. "
               f"Return only the reasoning text.")
    else:
        ask = (f"Context: {ctx}\nQuestion: {q}\nOptions: {answers}\n"
               f"The correct answer is '{gold}' (option {label}). Write a concise 1-2 sentence reasoning "
               f"that cites the specific evidence in the context supporting this answer. "
               f"Return only the reasoning text.")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You write concise, bias-free reasoning for VQA."},
                  {"role": "user", "content": ask}],
        temperature=0.3, max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def to_sharegpt(rec, reason, image_path):
    user = "<image>" + USER_TEMPLATE.format(
        context=rec["context"], question=rec["question"], options_block=format_options(rec["answers"]))
    label = int(rec["label"])
    target = f"<think>\n{reason}\n</think>\n{label}) {rec['answers'][label]}"
    return {"messages": [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target}], "images": [image_path]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--hf", default="walledai/BBQ")
    ap.add_argument("--n", type=int, default=5000, help="생성할 총 학습 샘플 수(균형)")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset
    from openai import OpenAI
    client = OpenAI(timeout=30, max_retries=2)  # 무한 hang 방지

    os.makedirs(args.out_dir, exist_ok=True)
    img = os.path.abspath(os.path.join(args.out_dir, "placeholder.jpg"))
    make_placeholder(img)

    # BBQ 로드 → ambig/disambig 균형 표본
    d = load_dataset(args.hf)
    pool = []
    for split in d.keys():
        for r in d[split]:
            ans = list(r["choices"])
            if len(ans) != 3:
                continue
            lab = int(r["answer"])
            pool.append({"context": r["context"], "question": r["question"], "answers": ans,
                         "label": lab, "label_type": "ambiguous" if is_unknown(ans[lab]) else "disambiguated"})
    amb = [x for x in pool if x["label_type"] == "ambiguous"]
    dis = [x for x in pool if x["label_type"] == "disambiguated"]
    random.shuffle(amb); random.shuffle(dis)
    half = args.n // 2
    items = amb[:half] + dis[:half]
    random.shuffle(items)

    # resume: 이미 처리된 인덱스 스킵
    cache = os.path.join(args.out_dir, "bbq_reason_cache.jsonl")
    done = 0
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            done = sum(1 for _ in f)
    todo = items[done:]
    print(f"총 {len(items)} 중 {done} 완료, {len(todo)} 생성 시작 (model={args.model})")

    counter = {"n": 0, "ok": 0, "err": 0}

    def work(rec):
        try:
            reason = gen_reason(client, args.model, rec["context"], rec["question"],
                                rec["answers"], rec["label"], rec["label_type"])
        except Exception as e:  # noqa: BLE001
            reason = None
            with _lock:
                counter["err"] += 1
                if counter["err"] <= 3:
                    print("ERR:", type(e).__name__, str(e)[:150], flush=True)
        sg = to_sharegpt(rec, reason, img) if reason else None
        with _lock:
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps(sg, ensure_ascii=False) + "\n" if sg else "null\n")
            counter["n"] += 1
            if sg:
                counter["ok"] += 1
            if counter["n"] % 25 == 0:
                print(f"PROGRESS {counter['n']}/{len(todo)} ok={counter['ok']} err={counter['err']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    # 캐시 → 최종 json
    records = []
    with open(cache, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj:
                records.append(obj)
    with open(os.path.join(args.out_dir, "bbq_reason_train.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump({"bbq_reasoning_real": {
            "file_name": "bbq_reason_train.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                     "assistant_tag": "assistant", "system_tag": "system"}}}, f, ensure_ascii=False, indent=2)
    print(f"완료: {len(records)}개 → {args.out_dir}/bbq_reason_train.json (dataset key: bbq_reasoning_real)")


if __name__ == "__main__":
    main()
