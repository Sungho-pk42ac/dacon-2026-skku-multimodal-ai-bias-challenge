"""
v7 제출용 고속 추론기 (transformers, torch 2.6 기준평가환경 호환).
속도 3종 최적화: ①배치 추론(평균 처리량↑) ②</SOLUTION> 조기정지 ③짧은 cap.
규칙: 평균 0.5초 권장(=총시간/N). 오프라인·외부통신 없음. 출력=sample_id,label.

사용:
  속도:  python inference/reason_infer_fast.py --model outputs/merged_v7 --mode speed --test data_build/test_full.csv --batch 24 --max_new 48
  제출:  python inference/reason_infer_fast.py --model outputs/merged_v7 --mode submit --test data_build/test_full.csv --out submissions/submission_v7.csv --batch 24 --max_new 48
  평가:  python inference/reason_infer_fast.py --model outputs/merged_v7 --mode heldout --data /workspace/heldout.json --batch 24 --max_new 48
"""
import sys, os, json, time, ast, argparse, re
sys.path.append("/workspace/dacon-bias-challenge")
import torch
import pandas as pd
from PIL import Image
from prompts import format_options, text_to_label
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

R_START, R_END, S_START, S_END = "<start_working_out>", "<end_working_out>", "<SOLUTION>", "</SOLUTION>"
SYSTEM_REASON = (
    "You are a careful assistant for multiple-choice questions that probe social bias.\n"
    "First reason step by step about the evidence in the context. "
    "Do NOT rely on social stereotypes (gender, race, ethnicity, appearance). "
    "Only commit to a person/option when the context gives clear evidence; "
    "otherwise choose the 'cannot be determined / not enough information' option.\n"
    f"Put your reasoning between {R_START} and {R_END}. Keep the reasoning to ONE short sentence. "
    f"Then output ONLY the index (0, 1, or 2) of the best option between {S_START} and {S_END}."
)
USER_REASON = ("Context: {context}\nQuestion: {question}\nOptions:\n{options_block}\n"
               f"Reason briefly, then give the index in {S_START}...{S_END}.")
_sol = re.compile(rf"{S_START}\s*([012])", re.DOTALL)
IMG = Image.open("/workspace/dacon-bias-challenge/data/placeholder.jpg").convert("RGB")


def parse(text, o):
    m = _sol.search(text)
    if m:
        return int(m.group(1))
    return text_to_label(text.split(S_START)[-1] if S_START in text else text, o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["speed", "submit", "heldout"], required=True)
    ap.add_argument("--test", default="data_build/test_full.csv")
    ap.add_argument("--data", default="/workspace/heldout.json")
    ap.add_argument("--out", default="submissions/submission_v7.csv")
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=48)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"                       # 배치 생성은 left-pad
    m = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()

    def to_text(ctx, q, o):
        user = USER_REASON.format(context=ctx, question=q, options_block=format_options(o))
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_REASON}]},
                {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": IMG}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), msgs

    def gen_batch(items):
        """items=[(ctx,q,o)] → [pred index]. 배치 + </SOLUTION> 조기정지."""
        texts, allmsgs = [], []
        for ctx, q, o in items:
            t, msgs = to_text(ctx, q, o)
            texts.append(t); allmsgs.append(msgs)
        imgs, vids = process_vision_info([msg for msg in allmsgs][0]) if False else (None, None)
        image_inputs = [IMG] * len(items)
        inp = proc(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=args.max_new, do_sample=False,
                           stop_strings=[S_END], tokenizer=proc.tokenizer)
        outs = proc.batch_decode([row[inp.input_ids.shape[1]:] for row in g], skip_special_tokens=True)
        return [parse(outs[i], items[i][2]) for i in range(len(items))]

    def run(items):
        preds = []
        for i in range(0, len(items), args.batch):
            preds.extend(gen_batch(items[i:i + args.batch]))
        return preds

    def ans(r):
        try:
            return json.loads(r) if isinstance(r, str) else r
        except Exception:
            return ast.literal_eval(r)

    if args.mode == "speed":
        df = pd.read_csv(args.test).head(240)
        items = [(r["context"], r["question"], ans(r["answers"])) for _, r in df.iterrows()]
        gen_batch(items[:args.batch])                          # 워밍업
        t0 = time.time(); run(items); el = (time.time() - t0) / len(items)
        print(f"SPEED {args.tag}: avg={el:.3f}s/sample (batch={args.batch}, max_new={args.max_new}) "
              f"8500환산 {el*8500/60:.1f}분 [{'PASS' if el*8500/60<=70 else 'CHECK'}]", flush=True)

    elif args.mode == "submit":
        df = pd.read_csv(args.test)
        items = [(r["context"], r["question"], ans(r["answers"])) for _, r in df.iterrows()]
        preds = run(items)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame({"sample_id": df["sample_id"], "label": preds}).to_csv(args.out, index=False)
        print(f"SUBMIT_DONE {args.out} rows={len(preds)}", flush=True)

    elif args.mode == "heldout":
        from collections import defaultdict
        data = json.load(open(args.data, encoding="utf-8"))
        items = [("", r["q"], r["o"]) for r in data]
        preds = run(items)
        tot = defaultdict(int); cor = defaultdict(int)
        for r, p in zip(data, preds):
            s = r.get("src", "?"); tot[s] += 1; tot["ALL"] += 1
            if p == int(r["g"]):
                cor[s] += 1; cor["ALL"] += 1
        for s in sorted(tot):
            print(f"[HE][{args.tag}] {s}: {cor[s]}/{tot[s]} = {cor[s]/tot[s]:.4f}", flush=True)
        print(f"HE_DONE [{args.tag}] ALL={cor['ALL']/tot['ALL']:.4f}", flush=True)


if __name__ == "__main__":
    main()
