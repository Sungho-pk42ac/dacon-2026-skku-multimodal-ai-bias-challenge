# held-out OOD 평가: 모델 1개를 heldout.json(채굴 시 떼어둔 미학습 문항)으로 채점.
# v4/v5/v6를 *동일* held-out으로 돌려 비교 → 정확도 상승=일반화 개선(암기 아님) 증거.
# 사용: python inference/eval_heldout.py --model outputs/merged_v6 --data /workspace/heldout.json
import sys, os, json, argparse
sys.path.append("/workspace/dacon-bias-challenge")
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/workspace/heldout.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or os.path.basename(args.model.rstrip("/"))

    img = Image.open("/workspace/dacon-bias-challenge/data/placeholder.jpg").convert("RGB")
    data = json.load(open(args.data, encoding="utf-8"))
    m = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(args.model)

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

    from collections import defaultdict
    tot = defaultdict(int); cor = defaultdict(int)
    for r in data:
        p = pred(r["q"], r["o"])
        s = r.get("src", "?")
        tot[s] += 1; tot["ALL"] += 1
        if p == int(r["g"]):
            cor[s] += 1; cor["ALL"] += 1
    print(f"=== HELDOUT EVAL [{tag}] ({args.data}) ===", flush=True)
    for s in sorted(tot):
        print(f"[HE][{tag}] {s}: {cor[s]}/{tot[s]} = {cor[s]/tot[s]:.4f}", flush=True)
    print(f"HE_DONE [{tag}] ALL={cor['ALL']/tot['ALL']:.4f}", flush=True)


if __name__ == "__main__":
    main()
