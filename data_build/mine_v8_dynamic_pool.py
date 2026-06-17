"""
v8 오프라인 동적샘플링 (Phase 7, DAPO dynamic sampling의 오프라인판).
온라인 동적샘플링이 복잡하므로, 학습 전 v8_pool을 사전 정제:
 각 프롬프트를 모델로 G회 샘플링 → 정답수 k 집계.
   k==0 too_hard(advantage 0), k==G too_easy(advantage 0), 0<k<G useful(학습신호 有).
 useful 위주로 남기고 too_hard/too_easy 통계 로깅.
입력 /workspace/v8_pool.json → 출력 /workspace/v8_pool_dynamic.json + 통계.
누출방지: DACON 평가셋 미사용(v8_pool은 공개데이터 기반).

사용: python data_build/mine_v8_dynamic_pool.py --model outputs/merged_v4 --pool /workspace/v8_pool.json --G 8
"""
import argparse, json, os, sys
sys.path.append("/workspace/dacon-bias-challenge")
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
from transformers import AutoProcessor, AutoModelForImageTextToText
PH = Image.open("/workspace/dacon-bias-challenge/data/placeholder.jpg").convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/merged_v4")
    ap.add_argument("--pool", default="/workspace/v8_pool.json")
    ap.add_argument("--out", default="/workspace/v8_pool_dynamic.json")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--keep_extremes", action="store_true", help="too_hard/too_easy도 일부 유지")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model); proc.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                    device_map="auto", attn_implementation="sdpa").eval()
    pool = json.load(open(args.pool, encoding="utf-8"))

    def k_correct(it):
        user = USER_TEMPLATE.format(context=it.get("context", ""), question=it["question"],
                                    options_block=format_options(it["answers"]))
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": PH}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t] * args.G, images=[PH] * args.G, padding=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=4, do_sample=True, temperature=0.9, top_p=0.95)
        outs = proc.batch_decode([r[inp.input_ids.shape[1]:] for r in g], skip_special_tokens=True)
        return sum(1 for o in outs if text_to_label(o, it["answers"]) == int(it["label"]))

    useful, too_hard, too_easy = [], 0, 0
    for i, it in enumerate(pool):
        k = k_correct(it)
        it.setdefault("meta", {})["dyn_k"] = k; it["meta"]["dyn_G"] = args.G
        if k == 0:
            too_hard += 1
            if args.keep_extremes: useful.append(it)
        elif k == args.G:
            too_easy += 1
            if args.keep_extremes: useful.append(it)
        else:
            useful.append(it)
        if (i + 1) % 100 == 0:
            print(f"[DYN] {i+1}/{len(pool)} useful={len(useful)} hard={too_hard} easy={too_easy}", flush=True)
    json.dump(useful, open(args.out, "w"), ensure_ascii=False)
    stats = {"input": len(pool), "useful": len(useful), "too_hard": too_hard, "too_easy": too_easy, "G": args.G}
    json.dump(stats, open("/workspace/v8_dynamic_stats.json", "w"), ensure_ascii=False, indent=2)
    print("DYN_STATS", json.dumps(stats), flush=True)
    print("MINE_V8_DYN_DONE", flush=True)


if __name__ == "__main__":
    main()
