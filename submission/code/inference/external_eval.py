"""
외부검증 평가기 (eval-only, 학습/파인튜닝 일절 없음).
독립 공개 MC셋(ood2: MMLU/HellaSwag/ARC-Challenge/WinoGrande)으로 v6/v8a/v8b/base 순수 일반화 측정.
- label은 모델 생성 텍스트에서만 파싱(외부API 0).
- 텍스트 전용 셋이므로 이미지는 placeholder 사용(사용자 이미지규칙 준수).
- 이미지 ablation: placeholder vs blank(흰배경)로 'image-invariance'(무관 이미지에 흔들리지 않는가) 측정.
- 옵션순서 셔플 일관성도 측정.
결과 누적: outputs/external_validation_results.json[dataset][tag], 표/ablation 파일은 별도 집계 스크립트로 생성.

사용: python inference/external_eval.py --model /workspace/models/v6 --tag v6 --dataset ood2 --data /workspace/ood2_eval.json
"""
import argparse, json, os, sys, time
sys.path.append("/workspace/dacon-bias-challenge")
import torch
from PIL import Image
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
from transformers import AutoProcessor, AutoModelForImageTextToText

PH = Image.open("/workspace/dacon-bias-challenge/data/placeholder.jpg").convert("RGB")
BLANK = Image.new("RGB", PH.size, (255, 255, 255))   # 흰배경 ablation 이미지
RES_JSON = "/workspace/dacon-bias-challenge/outputs/external_validation_results.json"


def shuffle_perm(ans, label, seed):
    import random as _r
    rnd = _r.Random(seed)
    idx = list(range(len(ans)))
    rnd.shuffle(idx)
    return [ans[i] for i in idx], idx.index(label), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="ood2")
    ap.add_argument("--data", default="/workspace/ood2_eval.json")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=4, help="단일토큰 파인튜닝 모델=4. 비파인튜닝 base는 자유텍스트라 24 권장.")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    m = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa").eval()
    data = json.load(open(args.data, encoding="utf-8"))

    def predict(triples, img):
        """triples=[(ctx,q,ans)], img=모든 샘플에 쓸 PIL 이미지. → (preds, raws, elapsed)."""
        preds, raws = [], []
        t0 = time.time()
        for i in range(0, len(triples), args.batch):
            ch = triples[i:i + args.batch]
            texts = []
            for ctx, q, ans in ch:
                user = USER_TEMPLATE.format(context=ctx, question=q, options_block=format_options(ans))
                msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                        {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": img}]}]
                texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            inp = proc(text=texts, images=[img] * len(ch), padding=True, return_tensors="pt").to(m.device)
            with torch.no_grad():
                g = m.generate(**inp, max_new_tokens=args.max_new, do_sample=False)
            outs = proc.batch_decode([r[inp.input_ids.shape[1]:] for r in g], skip_special_tokens=True)
            raws.extend(outs)
            preds.extend(text_to_label(outs[j], ch[j][2]) for j in range(len(ch)))
        return preds, raws, time.time() - t0

    base_triples = [(r.get("context", ""), r["question"], r["answers"]) for r in data]
    # 1) placeholder 이미지 예측(주 평가)
    p_ph, raw_ph, elapsed = predict(base_triples, PH)
    # 2) blank 이미지 예측(image-invariance ablation)
    p_blank, _r2, _e2 = predict(base_triples, BLANK)
    # 3) 옵션셔플 예측(순서 강건성)
    shuf = [shuffle_perm(r["answers"], r["label"], i) for i, r in enumerate(data)]
    p_shuf, _r3, _e3 = predict(
        [(data[i].get("context", ""), data[i]["question"], shuf[i][0]) for i in range(len(data))], PH)

    # ── 채점 ───────────────────────────────────────────────
    # TODO(user): 아래 per-source 집계와 셔플 일관성 판정 로직을 완성해주세요.
    # 제공 변수:
    #   data[i]: {"src","context","question","answers"(list),"label"(int),"unknown_idx"(여기선 None)}
    #   p_ph[i], p_blank[i], p_shuf[i]: 각 패스의 예측 인덱스(정수; 파싱실패 시 -1 등 비정상값 가능)
    #   shuf[i] = (shuffled_answers, shuffled_label, perm_idx)
    # 채워야 할 집계:
    #   per_src[src] = {"tot":N, "cor":정답수}   (정확도 = cor/tot, p_ph 기준)
    #   invariant     = (p_ph[i]==p_blank[i]) 인 샘플 수  → image_invariance = invariant/N
    #   shuf_consist  = '원본예측이 가리킨 옵션텍스트' == '셔플예측이 가리킨 옵션텍스트' 인 샘플 수
    #                    (인덱스가 아니라 '선택한 보기 내용'이 같아야 순서무관 일관성)
    from collections import defaultdict
    per_src = defaultdict(lambda: {"tot": 0, "cor": 0})
    invariant = 0
    shuf_consist = 0
    N = len(data)
    for i, r in enumerate(data):
        src, lab, ans = r["src"], int(r["label"]), r["answers"]
        # (1) 소스별 정확도 (placeholder 패스 기준)
        per_src[src]["tot"] += 1
        if p_ph[i] == lab:
            per_src[src]["cor"] += 1
        # (2) image-invariance: placeholder vs blank 예측 일치
        if p_ph[i] == p_blank[i]:
            invariant += 1
        # (3) 셔플 일관성: '고른 보기 내용'이 순서무관하게 동일한가 (인덱스 아님)
        sem_ph = ans[p_ph[i]] if 0 <= p_ph[i] < len(ans) else None
        shuf_ans = shuf[i][0]
        sem_shuf = shuf_ans[p_shuf[i]] if 0 <= p_shuf[i] < len(shuf_ans) else None
        if sem_ph is not None and sem_ph == sem_shuf:
            shuf_consist += 1

    # ── 지표 산출 ─────────────────────────────────────────
    overall_cor = sum(v["cor"] for v in per_src.values())
    overall_tot = sum(v["tot"] for v in per_src.values())
    def _fmt_ok(t):
        t = str(t).strip()
        return len(t) > 0 and t[0] in ("0", "1", "2") and len(t) <= 2
    fmt_valid = sum(_fmt_ok(t) for t in raw_ph) / max(1, len(raw_ph))
    res = {
        "tag": args.tag, "dataset": args.dataset, "n": N,
        "accuracy": round(overall_cor / max(1, overall_tot), 4),
        "accuracy_by_src": {s: round(v["cor"] / max(1, v["tot"]), 4) for s, v in sorted(per_src.items())},
        "image_invariance_ph_vs_blank": round(invariant / max(1, N), 4),
        "shuffle_consistency": round(shuf_consist / max(1, N), 4),
        "output_format_validity": round(fmt_valid, 4),
        "inference_speed_sec_per_sample": round(elapsed / max(1, N), 4),
        "image_real_count": 0, "image_placeholder_count": N,   # 텍스트 전용 → 전량 placeholder
        "image_missing_count": 0,
    }
    print("EXT_EVAL_RESULT", json.dumps(res, ensure_ascii=False), flush=True)

    allr = {}
    if os.path.exists(RES_JSON):
        try:
            allr = json.load(open(RES_JSON, encoding="utf-8"))
        except Exception:
            allr = {}
    allr.setdefault(args.dataset, {})[args.tag] = res
    os.makedirs(os.path.dirname(RES_JSON), exist_ok=True)
    json.dump(allr, open(RES_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"EXT_EVAL_DONE [{args.dataset}/{args.tag}] -> {RES_JSON}", flush=True)


if __name__ == "__main__":
    main()
