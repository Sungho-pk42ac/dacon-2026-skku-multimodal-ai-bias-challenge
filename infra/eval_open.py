"""
오픈 held-out 평가 — v6가 보고된 성능을 재현하는지 확인 (DACON test 불필요, 전부 공개 데이터).

평가셋: make_bbq_clean.py가 생성하는 BBQ held-out (data/bbq_val_clean.jsonl, seed 42 결정론).
지표(논문 §5.1과 동일 정의):
  - BBQ ambiguous acc / disambiguated acc
  - BALANCED ACC = (amb + dis)/2          ← 보고치 1.0
  - 옵션셔플 일관성 (순서 바꿔도 같은 답)  ← 보고치 0.9544
라벨은 모델 생성 텍스트 파싱(text_to_label). greedy 디코딩. 외부 API 0.

사용:
  python infra/eval_open.py --model <v6경로 또는 HF> --data data/bbq_val_clean.jsonl
"""
import argparse
import json
import os
import sys
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/bbq_val_clean.jsonl")
    ap.add_argument("--limit", type=int, default=0, help=">0이면 앞 N개만(빠른 점검)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--image_mode", choices=["placeholder", "real"], default="placeholder",
                    help="placeholder=중립회색(텍스트집중) / real=공개 실사진(cifar10) 부착(실이미지 레짐)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # 이미지 풀 구성: placeholder=중립회색 1장 / real=공개 실사진(cifar10) 다수를 샘플마다 순환 부착
    if args.image_mode == "real":
        from datasets import load_dataset
        n_img = max(64, args.limit or 64)
        ds_img = load_dataset("uoft-cs/cifar10", split=f"test[:{n_img}]")
        image_pool = [r["img"].convert("RGB").resize((336, 336)) for r in ds_img]
        print(f"[eval] 실이미지 모드 — cifar10 실사진 {len(image_pool)}장 순환 부착", flush=True)
    else:
        image_pool = [Image.new("RGB", (336, 336), (127, 127, 127))]   # 중립 placeholder
        print("[eval] placeholder 모드 — 중립 회색 이미지(텍스트집중)", flush=True)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} samples · model={args.model}", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    print("[eval] 모델 로드 완료", flush=True)

    def predict(items, imgs):
        """items=[(ctx, q, answers)], imgs=샘플별 이미지 → [예측 라벨] (make_submission과 동일 경로)."""
        texts = []
        for (ctx, q, ans), im in zip(items, imgs):
            user = USER_TEMPLATE.format(context=ctx, question=q, options_block=format_options(ans))
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{"type": "text", "text": user},
                                                 {"type": "image", "image": im}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        inp = proc(text=texts, images=list(imgs), padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=4, do_sample=False)
        outs = proc.batch_decode([r[inp.input_ids.shape[1]:] for r in g], skip_special_tokens=True)
        return [text_to_label(o, it[2]) for o, it in zip(outs, items)]

    amb_t = amb_c = dis_t = dis_c = 0
    sh_t = sh_c = 0
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        # 샘플별 이미지(placeholder면 모두 동일, real이면 cifar10 순환) — 한 샘플엔 같은 이미지로 orig/shuffle
        imgs = [image_pool[(i + j) % len(image_pool)] for j in range(len(chunk))]

        # 1) 원본 순서 정확도
        items = [(r.get("context", ""), r["question"], r["answers"]) for r in chunk]
        preds = predict(items, imgs)

        # 2) 옵션 셔플 순서(순서불변 검증) — sample_id 시드로 결정론 셔플
        sh_items, perms = [], []
        for r in chunk:
            ans = r["answers"]
            idx = list(range(len(ans)))
            random.Random(str(r["sample_id"])).shuffle(idx)
            perms.append(idx)
            sh_items.append((r.get("context", ""), r["question"], [ans[j] for j in idx]))
        sh_preds = predict(sh_items, imgs)

        for r, p, sp, si in zip(chunk, preds, sh_preds, sh_items):
            amb = r.get("label_type") == "ambiguous"
            ok = (p == int(r["label"]))
            if amb:
                amb_t += 1; amb_c += ok
            else:
                dis_t += 1; dis_c += ok
            # 셔플 일관성: 원본 예측이 가리킨 '텍스트' == 셔플 예측이 가리킨 '텍스트'
            ans = r["answers"]
            sh_ans = si[2]   # 셔플된 선택지
            orig_text = ans[p] if 0 <= p < len(ans) else None
            sh_text = sh_ans[sp] if 0 <= sp < len(sh_ans) else None
            sh_t += 1
            sh_c += int(orig_text is not None and orig_text == sh_text)

        if (i // args.batch) % 5 == 0:
            print(f"  {min(i + args.batch, len(rows))}/{len(rows)}", flush=True)

    amb_acc = amb_c / max(1, amb_t)
    dis_acc = dis_c / max(1, dis_t)
    balanced = (amb_acc + dis_acc) / 2
    shuffle_cons = sh_c / max(1, sh_t)
    dt = time.time() - t0

    print("\n========== 결과 (오픈 held-out, BBQ val) ==========")
    print(f"ambiguous acc     : {amb_acc:.4f}   (n={amb_t})")
    print(f"disambiguated acc : {dis_acc:.4f}   (n={dis_t})")
    print(f"BALANCED ACC      : {balanced:.4f}   <-- 보고치 1.0")
    print(f"옵션셔플 일관성   : {shuffle_cons:.4f}   <-- 보고치 0.9544")
    print(f"추론 속도         : {dt / max(1,len(rows)) * 1000 / 2:.0f}ms/샘플 (배치 평균)")
    print(f"EVAL_OPEN_DONE balanced={balanced:.4f} shuffle={shuffle_cons:.4f}")


if __name__ == "__main__":
    main()
