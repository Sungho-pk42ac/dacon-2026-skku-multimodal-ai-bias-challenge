"""
오프라인 추론 → 제출 CSV 생성 (추론 전용 — 학습 코드와 분리).

대회 확정 포맷:
  test.csv 컬럼 = sample_id, image_path, context, question, answers(3개 선택지 JSON 문자열)
  제출 = sample_submission.csv 형식: sample_id, label(0/1/2)

규칙 준수:
  * 외부 API/인터넷 호출 0 (로컬 가중치, vLLM 오프라인).
  * 최종 답은 LLM 생성 텍스트 → 3개 옵션 중 인덱스로 매핑(정규화만, 결정은 LLM).
  * greedy + 짧은 max_tokens 로 샘플당 평균 0.5초 목표 (test 8,500개 ≈ 70분).
"""

import argparse
import ast
import json
import logging
import os
import sys
import time

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_answers(raw):
    """answers 컬럼(JSON 문자열) → 리스트. JSON 실패 시 파이썬 리터럴로 폴백."""
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(raw)


def resolve_image_path(images_dir, image_path, fallback=None):
    """test.csv의 image_path(예: ./images/xxx.jpg)를 실제 경로로 변환. 없으면 fallback."""
    base = os.path.basename(str(image_path))
    cand = os.path.join(images_dir, base)
    if os.path.exists(cand):
        return cand
    alt = os.path.join(images_dir, "..", str(image_path).lstrip("./"))
    if os.path.exists(alt):
        return alt
    if fallback and os.path.exists(fallback):
        return fallback  # 실제 이미지 없을 때(형식 스모크 테스트 등)
    return cand


def build_input(processor, context, question, answers, image):
    """messages → vLLM 입력(prompt + multimodal)."""
    user_text = USER_TEMPLATE.format(
        context=context, question=question, options_block=format_options(answers)
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image", "image": image},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return {"prompt": text, "multi_modal_data": {"image": image_inputs}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="merged 모델 경로 또는 모델 ID")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--images_dir", default="data/test/images")
    parser.add_argument("--out", default="submissions/sub.csv")
    parser.add_argument("--fallback_image", default="data/placeholder.jpg",
                        help="실제 이미지 없을 때 대체(형식 스모크 테스트용)")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    logger.info("vLLM 모델 로드: %s", args.model)
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16")
    sampling = SamplingParams(temperature=0.0, max_tokens=24)  # greedy + 짧게

    df = pd.read_csv(args.test)
    logger.info("평가 문항 수: %d", len(df))

    inputs, metas = [], []
    for _, row in df.iterrows():
        answers = parse_answers(row["answers"])
        img = Image.open(resolve_image_path(
            args.images_dir, row["image_path"], args.fallback_image)).convert("RGB")
        inputs.append(build_input(processor, row["context"], row["question"], answers, img))
        metas.append((row["sample_id"], answers))

    t0 = time.time()
    outputs = llm.generate(inputs, sampling)
    elapsed = time.time() - t0
    logger.info("추론 %d건 / %.1fs / 샘플당 %.3fs (기준 0.5s)",
                len(inputs), elapsed, elapsed / max(len(inputs), 1))

    rows = []
    for (sample_id, answers), out in zip(metas, outputs):
        label = text_to_label(out.outputs[0].text, answers)
        rows.append({"sample_id": sample_id, "label": label})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")
    logger.info("제출 파일 저장 → %s", args.out)


if __name__ == "__main__":
    main()
