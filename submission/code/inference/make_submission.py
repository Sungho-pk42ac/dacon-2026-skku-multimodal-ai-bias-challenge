"""
DACON 제출 CSV 생성 (추론 전용 — 학습과 분리). transformers 기반(외부 API 0, torch 2.6 기준환경 호환).

이미지 규칙(중요):
 - DACON 최종추론/제출은 test.csv의 image_path + --image_root로 **실제 테스트 이미지**를 로드한다.
 - placeholder(고정 더미)는 DACON 테스트/제출 추론에 사용 금지.
 - real_image_loaded_count / placeholder_fallback_count를 검증 JSON에 기록.
 - placeholder_fallback_count > 0 이면 **중단**하고 제출을 INVALID로 표시(--allow_placeholder로만 허용).
 - 최종 라벨은 모델 생성 텍스트에서 파싱(text_to_label) — 규칙기반 정답선택 아님.

산출:
 - submissions/submission_v8b.csv          (sample_submission과 컬럼·행순서 동일)
 - submissions/preview_v8b.jsonl           (행별 모델 원출력/파싱라벨/이미지소스)
 - submissions/submission_v8b_validation.json (검증 결과 + 이미지 카운트)

실행:
  python inference/make_submission.py --model outputs/merged_v8b \
    --test_csv data/test.csv --sample_submission data/sample_submission.csv \
    --image_root data/test_images --out submissions/submission_v8b.csv \
    --batch_size 8 --max_new_tokens 4 --seed 42
"""
import argparse, ast, json, logging, os, sys
import pandas as pd
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_answers(raw):
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw)


def resolve_image_path(image_root, image_path, test_csv_dir):
    """실제 테스트 이미지 경로 해석. 존재하면 경로, 없으면 None(=fallback 대상)."""
    ip = str(image_path)
    cands = []
    if image_root:
        cands.append(os.path.join(image_root, os.path.basename(ip)))
        cands.append(os.path.join(image_root, ip.lstrip("./")))
    cands.append(os.path.join(test_csv_dir, ip.lstrip("./")))   # test.csv 기준 상대경로
    cands.append(ip)
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--sample_submission", default="data/sample_submission.csv")
    ap.add_argument("--image_root", default="data/test_images")
    ap.add_argument("--out", default="submissions/submission_v8b.csv")
    ap.add_argument("--preview", default=None, help="기본: out 경로 기반 preview_*.jsonl")
    ap.add_argument("--validation", default=None, help="기본: out 경로 기반 *_validation.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pixels", type=int, default=0,
                    help="0(기본)=원본 해상도 그대로(심사환경과 동일·충실). >0이면 vision 토큰/메모리 상한 캡(속도/OOM 회피용, 입력 변경됨).")
    ap.add_argument("--allow_placeholder", action="store_true",
                    help="실제 이미지 없을 때 placeholder 허용(기본 금지). 켜도 검증에 카운트 기록.")
    ap.add_argument("--fallback_image", default="data/placeholder.jpg")
    args = ap.parse_args()

    import random
    # 재현성: 시드 고정 (추론은 greedy라 결정론적이지만 안전하게 함께 고정)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = os.path.splitext(os.path.basename(args.out))[0]
    sub_dir = os.path.dirname(args.out) or "."
    preview_path = args.preview or os.path.join(sub_dir, f"preview_{tag.replace('submission_', '')}.jsonl")
    valid_path = args.validation or os.path.join(sub_dir, f"{tag}_validation.json")
    os.makedirs(sub_dir, exist_ok=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText

    logger.info("모델 로드: %s", args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    # 기본: 모델 processor 기본 해상도 그대로(원본 충실, 심사환경과 동일). --max_pixels>0 일 때만 캡 적용.
    if args.max_pixels and args.max_pixels > 0:
        try:
            processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
        except TypeError:
            processor = AutoProcessor.from_pretrained(args.model)
        try:
            if hasattr(processor, "image_processor"):
                processor.image_processor.max_pixels = args.max_pixels
        except Exception as e:
            logger.warning("max_pixels 설정 실패(무시): %s", str(e)[:60])
        logger.info("max_pixels 캡 적용: %d", args.max_pixels)
    else:
        processor = AutoProcessor.from_pretrained(args.model)
        logger.info("max_pixels 미적용(원본 해상도, processor 기본)")
    processor.tokenizer.padding_side = "left"

    df = pd.read_csv(args.test_csv)
    sample = pd.read_csv(args.sample_submission)
    test_dir = os.path.dirname(os.path.abspath(args.test_csv))
    logger.info("test 행수: %d / sample_submission 행수: %d", len(df), len(sample))

    # sample_submission 컬럼 구조 파악(컬럼·순서 정확히 일치)
    sub_cols = list(sample.columns)
    id_col = sub_cols[0]
    label_col = sub_cols[-1]

    # ---- 이미지 해석 + 실제 로드 카운트 ----
    real_cnt = 0           # 실제 테스트 이미지를 로드한 행 수
    fb_cnt = 0             # placeholder로 대체된(fallback) 행 수
    items = []             # (sample_id, context, question, answers, image, img_source)
    fb_img = None          # placeholder 이미지(최초 1회만 로드해 재사용)
    for _, r in df.iterrows():
        answers = parse_answers(r["answers"])
        p = resolve_image_path(args.image_root, r.get("image_path", ""), test_dir)
        if p is not None:
            try:
                img = Image.open(p).convert("RGB")
                real_cnt += 1
                src = p
            except Exception:
                p = None   # 손상 이미지 → fallback 대상
        if p is None:
            fb_cnt += 1
            src = "PLACEHOLDER"
            if fb_img is None:
                if args.fallback_image and os.path.exists(args.fallback_image):
                    fb_img = Image.open(args.fallback_image).convert("RGB")
                else:
                    # 미동봉/미지정 시 중립(회색) 이미지 자동생성 — 텍스트집중 레짐 결정론 유지(FileNotFoundError 방지)
                    fb_img = Image.new("RGB", (336, 336), (128, 128, 128))
            img = fb_img
        items.append((r["sample_id"], r.get("context", ""), r["question"], answers, img, src))

    logger.info("이미지: real=%d placeholder_fallback=%d", real_cnt, fb_cnt)

    # 규칙: 제출 추론에 placeholder fallback이 있으면 중단(허용 플래그 없을 때)
    if fb_cnt > 0 and not args.allow_placeholder:
        val = {"status": "INVALID_PLACEHOLDER_USED", "reason":
               "DACON 제출 추론에서 실제 이미지를 찾지 못해 placeholder fallback 발생",
               "real_image_loaded_count": real_cnt, "placeholder_fallback_count": fb_cnt,
               "test_rows": len(df), "image_root": args.image_root,
               "action_required": "실제 테스트 이미지를 --image_root에 배치 후 재생성"}
        json.dump(val, open(valid_path, "w"), ensure_ascii=False, indent=2)
        logger.error("STOP: placeholder_fallback_count=%d > 0 → 제출 INVALID. 검증: %s", fb_cnt, valid_path)
        sys.exit(2)

    # ---- 배치 추론 ----
    def gen_batch(batch_items):
        texts, imgs = [], []
        for _, ctx, q, ans, im, _src in batch_items:
            user = USER_TEMPLATE.format(context=ctx, question=q, options_block=format_options(ans))
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{"type": "text", "text": user}, {"type": "image", "image": im}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.append(im)
        inp = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        return processor.batch_decode([row[inp.input_ids.shape[1]:] for row in g], skip_special_tokens=True)

    preds = {}          # sample_id → 예측 라벨(0/1/2)
    previews = []       # 행별 원출력·파싱결과 기록
    fmt_valid = 0       # 단일토큰 형식(0/1/2로 시작) 유효 건수
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        outs = gen_batch(chunk)
        for (sid_, ctx, q, ans, im, src), raw in zip(chunk, outs):
            label = text_to_label(raw, ans)          # 최종 라벨 = 모델 생성 텍스트 파싱
            t = str(raw).strip()
            if len(t) > 0 and t[0] in ("0", "1", "2"):
                fmt_valid += 1
            preds[sid_] = int(label)
            previews.append({"sample_id": sid_, "raw_output": raw, "parsed_label": int(label),
                             "image_source": src})
        if (i + args.batch_size) % 400 == 0:
            logger.info("%d/%d", min(i + args.batch_size, len(items)), len(items))

    # ---- sample_submission 순서·컬럼으로 출력 ----
    out_df = sample.copy()
    missing = [sid_ for sid_ in sample[id_col] if sid_ not in preds]
    out_df[label_col] = [preds.get(sid_, -1) for sid_ in sample[id_col]]

    out_df.to_csv(args.out, index=False)
    with open(preview_path, "w", encoding="utf-8") as f:
        for p in previews:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    labels = out_df[label_col].tolist()
    val = {
        "status": "OK" if (not missing and all(l in (0, 1, 2) for l in labels)) else "INVALID",
        "model": args.model, "out": args.out,
        "row_count": len(out_df), "sample_submission_row_count": len(sample),
        "row_count_matches": len(out_df) == len(sample),
        "id_order_matches": out_df[id_col].tolist() == sample[id_col].tolist(),
        "labels_all_in_012": all(l in (0, 1, 2) for l in labels),
        "null_label_count": int(sum(1 for l in labels if l not in (0, 1, 2))),
        "missing_prediction_count": len(missing),
        "real_image_loaded_count": real_cnt,
        "placeholder_fallback_count": fb_cnt,
        "output_format_validity": round(fmt_valid / max(1, len(items)), 4),
        "columns": sub_cols, "seed": args.seed,
        "label_source": "model_generated_text(text_to_label)",
    }
    json.dump(val, open(valid_path, "w"), ensure_ascii=False, indent=2)
    logger.info("제출 저장 → %s (%d행) | 검증 status=%s | real=%d fb=%d fmt=%.4f",
                args.out, len(out_df), val["status"], real_cnt, fb_cnt, val["output_format_validity"])
    print("MAKE_SUBMISSION_DONE", json.dumps({k: val[k] for k in
          ("status", "row_count", "real_image_loaded_count", "placeholder_fallback_count",
           "output_format_validity")}, ensure_ascii=False), flush=True)
    if val["status"] != "OK":
        sys.exit(3)


if __name__ == "__main__":
    main()
