"""
통합 추론기 (Phase 4). v4/v6/v7/v8 공용. torch 2.6 기준환경 호환(transformers, 외부API 0).
- 출력형식 자동: --fmt single(0/1/2) | evidence(<SOLUTION>idx). 최종 label은 *모델 생성 텍스트*에서만 도출.
- 실이미지: image_path 있으면 로드, 없거나 깨지면 placeholder fallback.
- 모드: speed | submit | eval | ablation. submit 출력은 항상 sample_id,label.
- 배치 + </SOLUTION> 조기정지(evidence)로 속도 최적화.

사용 예:
  submit:  python inference/infer_unified.py --model outputs/merged_v8a --mode submit --fmt single \
             --test data_build/test_full.csv --images_dir data/test/images --out submissions/submission_v8a.csv
  eval:    python inference/infer_unified.py --model outputs/merged_v8a --mode eval --fmt single --data /workspace/v8_eval.json
  speed:   python inference/infer_unified.py --model outputs/merged_v8a --mode speed --fmt single --test data_build/test_full.csv
  ablation:python inference/infer_unified.py --model outputs/merged_v8a --mode ablation --fmt single --data /workspace/v8_eval.json
"""
import argparse, ast, json, os, re, sys, time
sys.path.append("/workspace/dacon-bias-challenge")
import torch, pandas as pd
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
from transformers import AutoProcessor, AutoModelForImageTextToText

R_START, R_END, S_START, S_END = "<start_working_out>", "<end_working_out>", "<SOLUTION>", "</SOLUTION>"
SYS_EVID = (SYSTEM_MESSAGE + f"\nBriefly state evidence between {R_START} and {R_END} (one short clause), "
            f"then the index between {S_START} and {S_END}.")
_solre = re.compile(rf"{S_START}\s*([012])", re.DOTALL)
PLACEHOLDER = "/workspace/dacon-bias-challenge/data/placeholder.jpg"


def load_img(path):
    if path:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    return Image.open(PLACEHOLDER).convert("RGB")


def parse_label(text, answers, fmt):
    if fmt == "evidence":
        m = _solre.search(text)
        if m: return int(m.group(1))
        text = text.split(S_START)[-1] if S_START in text else text
    return text_to_label(text, answers)


def parse_answers(raw):
    if isinstance(raw, list): return raw
    try: return json.loads(raw)
    except Exception: return ast.literal_eval(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["speed", "submit", "eval", "ablation"], required=True)
    ap.add_argument("--fmt", choices=["single", "evidence"], default="single")
    ap.add_argument("--test", default="data_build/test_full.csv")
    ap.add_argument("--data", default="/workspace/v8_eval.json")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--out", default="submissions/submission_v8.csv")
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    max_new = 4 if args.fmt == "single" else 48
    sysmsg = SYS_EVID if args.fmt == "evidence" else SYSTEM_MESSAGE

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    stop = [S_END] if args.fmt == "evidence" else None

    def gen_batch(items):
        """items=[(context,question,answers,image)] → [텍스트]."""
        texts, imgs = [], []
        for ctx, q, ans, im in items:
            user = USER_TEMPLATE.format(context=ctx, question=q, options_block=format_options(ans))
            msgs = [{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
                    {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": im}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)); imgs.append(im)
        inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to(m.device)
        kw = dict(max_new_tokens=max_new, do_sample=False)
        if stop: kw.update(stop_strings=stop, tokenizer=proc.tokenizer)
        with torch.no_grad():
            g = m.generate(**inp, **kw)
        return proc.batch_decode([row[inp.input_ids.shape[1]:] for row in g], skip_special_tokens=True)

    def run(items):
        preds = []
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            outs = gen_batch(chunk)
            preds.extend(parse_label(outs[j], chunk[j][2], args.fmt) for j in range(len(chunk)))
        return preds

    if args.mode in ("submit", "speed"):
        df = pd.read_csv(args.test)
        if args.mode == "speed": df = df.head(240)
        def img_for(r):
            p = None
            if args.images_dir and "image_path" in r and isinstance(r["image_path"], str):
                cand = os.path.join(args.images_dir, os.path.basename(r["image_path"]))
                p = cand if os.path.exists(cand) else None
            return load_img(p)
        items = [(r.get("context", ""), r["question"], parse_answers(r["answers"]), img_for(r)) for _, r in df.iterrows()]
        if args.mode == "speed":
            gen_batch(items[:args.batch])  # warmup
            t0 = time.time(); run(items); el = (time.time() - t0) / len(items)
            print(f"SPEED {args.tag}: avg={el:.3f}s/sample 8500환산 {el*8500/60:.1f}분 "
                  f"[{'PASS' if el*8500/60<=70 else 'CHECK'}]", flush=True)
        else:
            preds = run(items)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            pd.DataFrame({"sample_id": df["sample_id"], "label": preds}).to_csv(args.out, index=False)
            print(f"SUBMIT_DONE {args.out} rows={len(preds)}", flush=True)

    elif args.mode == "eval":
        from collections import defaultdict
        data = json.load(open(args.data, encoding="utf-8"))
        items = [(r.get("context", ""), r["question"], r["answers"], load_img(r.get("image_path"))) for r in data]
        preds = run(items)
        tot = defaultdict(int); cor = defaultdict(int)
        for r, p in zip(data, preds):
            s = r.get("src", "?"); tot[s] += 1; tot["ALL"] += 1
            if p == int(r["label"]): cor[s] += 1; cor["ALL"] += 1
        for s in sorted(tot):
            print(f"[EVAL][{args.tag}] {s}: {cor[s]}/{tot[s]} = {cor[s]/tot[s]:.4f}", flush=True)
        print(f"EVAL_DONE [{args.tag}] ALL={cor['ALL']/tot['ALL']:.4f}", flush=True)

    elif args.mode == "ablation":
        # 이미지 조건별 정확도: real / blank(placeholder) / shuffled(픽셀셔플)
        import numpy as np
        data = json.load(open(args.data, encoding="utf-8"))
        blank = Image.open(PLACEHOLDER).convert("RGB")
        def shuf(im):
            a = np.array(im); idx = np.random.permutation(a.shape[0]); return Image.fromarray(a[idx])
        for cond in ["real", "blank", "shuffled"]:
            items = []
            for r in data:
                base = load_img(r.get("image_path"))
                im = base if cond == "real" else (blank if cond == "blank" else shuf(base))
                items.append((r.get("context", ""), r["question"], r["answers"], im))
            preds = run(items)
            acc = sum(1 for r, p in zip(data, preds) if p == int(r["label"])) / max(1, len(data))
            print(f"ABLATION [{args.tag}] image={cond}: acc={acc:.4f}", flush=True)
        print(f"ABLATION_DONE [{args.tag}]", flush=True)


if __name__ == "__main__":
    main()
