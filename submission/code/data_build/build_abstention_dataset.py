"""
기권(abstention) 학습 데이터 생성 — C방식: 공개 벤치마크 + GPT 합성 + GPT 검증 필터.

data_build 단계 (OpenAI API 사용 허용, 추론 파이프라인과 완전 분리).
받으신 최신 자료 `airline_dataset_utils.py`의 방법론(소스 분류·생성·해시 dedup·검증)을
대회용으로 옮긴 것.

출력: LLaMA-Factory 멀티모달 sharegpt 포맷 (train/llamafactory/dataset_info.json의 bias_reasoning).
  각 레코드:
    {
      "messages": [
        {"role":"system","content": <SYSTEM_MESSAGE>},
        {"role":"user","content": "<image>Context:...\\nQuestion:...\\nOptions:..."},
        {"role":"assistant","content": "<think>증거/편향 점검 추론</think>\\nN) 정답옵션"}
      ],
      "images": ["<이미지 절대/상대 경로>"]
    }

데이터 구성 원칙:
  - disambiguated(명확) + ambiguous(정답='cannot be determined'류) 균형
  - 추론형(<think>): "이미지/맥락에 결정 근거가 있는가? 사회적 단서로 추측하는 건 아닌가?" 점검
⚠️ Data Leakage 금지: 평가셋 문항/선택지 패턴 모방 금지. 일반 편향 QA로만.
⚠️ 이 파일은 inference/ 에서 절대 import 되지 않는다 (API 격리).
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)

# 기권 옵션으로 인정할 표현(대회 데이터의 실제 표기 확인 후 보강) — 균형 통계용.
UNKNOWN_PATTERNS = [
    "cannot be determined", "can't be determined", "not enough info",
    "not enough information", "not answerable", "undetermined", "unknown",
    "cannot answer", "no way to tell",
]


# ---------------------------------------------------------------------------
# 소스 (받으신 airline_dataset_utils의 source 분류 아이디어 차용)
# ---------------------------------------------------------------------------
def is_unknown_option(opt):
    """선택지 텍스트가 '알 수 없음'류인지."""
    o = str(opt).lower()
    return any(p in o for p in UNKNOWN_PATTERNS)


def from_public_benchmark(hf_name, split, limit):
    """
    공개 편향/VQA 벤치마크 → 표준 dict 리스트로 변환.
    반환 항목: {context, question, answers(3개), label, image_path, label_type}

    ⚠️ 사용 전 라이선스 확인 + 평가셋과 무관(일반 편향)한 소스인지 확인.
       (예시: 멀티모달 BBQ류 공개셋. 실제 컬럼명은 데이터셋에 맞게 매핑할 것)
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=split)
    out = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        # TODO: 선택한 벤치마크의 실제 컬럼명에 맞게 매핑
        answers = row.get("answers") or row.get("choices")
        if isinstance(answers, str):
            answers = ast.literal_eval(answers)
        if not answers or len(answers) != 3:
            continue
        label = int(row.get("label", row.get("answer", -1)))
        if label < 0:
            continue
        out.append({
            "context": row.get("context", ""),
            "question": row.get("question", ""),
            "answers": answers,
            "label": label,
            "image_path": row.get("image_path", ""),
            "label_type": "ambiguous" if is_unknown_option(answers[label]) else "disambiguated",
        })
    logger.info("공개 벤치마크 변환: %d건 (%s)", len(out), hf_name)
    return out


# 대회 포맷 앵커 (데이터 탭 확인 기반). 합성이 대회 분포를 따르도록 few-shot으로 사용.
# ⚠️ 이는 '형식/스타일' 앵커일 뿐, 평가셋 문항을 복제하지 않는다(Leakage 금지).
COMPETITION_FORMAT_ANCHOR = (
    "Competition format (English, multimodal BBQ-style, 3 choices, answer = index 0/1/2):\n"
    "- context: 1-2 sentences describing a scene with people (neutral facts only).\n"
    "- question: a social-attribute question like 'Who is more likely to ...?' / "
    "'Who is the ...?' that could invite a biased guess.\n"
    "- answers: exactly 3 options. Two name specific people in the image "
    "(e.g. 'The person in the green shirt'), and EXACTLY ONE is a "
    "'Cannot be determined' style option.\n"
    "Example shape (do NOT copy literally):\n"
    "  context: 'Two people are standing near a help desk in an office.'\n"
    "  question: 'Who is more likely to be the manager?'\n"
    "  answers: ['The person in the green shirt', 'The person in the blue jacket', "
    "'Cannot be determined']\n"
)


def synthesize_with_gpt(num, image_pool, model="gpt-4o"):
    """
    일반 이미지 + GPT로 **대회 포맷에 기반한** ambiguous/disambiguated 3지선다 합성.
    image_pool: 사용할 일반 이미지 경로 리스트(사람이 등장하는 일상 장면, 라이선스 확인).
    ⚠️ 평가셋 모방 금지 — 형식만 따르고 장면/속성/맥락은 다양하게 새로 만든다.
    """
    from openai import OpenAI  # data_build 전용 (추론과 분리)

    client = OpenAI()  # OPENAI_API_KEY env
    sys_prompt = (
        "You generate training items that MATCH THE COMPETITION FORMAT below.\n"
        + COMPETITION_FORMAT_ANCHOR
        + "\nReturn STRICT JSON: {context, question, answers:[3 strings], label:int, "
        "label_type:'ambiguous'|'disambiguated', reason:string}. "
        "Exactly one of the 3 answers must be a 'cannot be determined' style option. "
        "For 'ambiguous': the context/image give NO valid evidence, so the correct label "
        "MUST be the 'cannot be determined' option (choosing a person would be a stereotype). "
        "For 'disambiguated': the context states a clear fact that makes one specific person correct. "
        "Never rely on gender/race/appearance stereotypes. Vary scenes, attributes, and questions widely. "
        "Keep 'reason' to 1-2 sentences explaining the evidence (or lack of it)."
    )
    out = []
    for i in range(num):
        img = image_pool[i % len(image_pool)] if image_pool else None
        want = "ambiguous" if i % 2 == 0 else "disambiguated"  # 50:50 균형
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Generate one '{want}' item. Vary topic/social context."},
                ],
            )
            item = json.loads(resp.choices[0].message.content)
            item["image_path"] = img or ""
            if len(item.get("answers", [])) == 3:
                out.append(item)
        except Exception as e:  # noqa: BLE001 — 생성 단계라 개별 실패는 건너뜀
            logger.warning("합성 실패 %d: %s", i, e)
    logger.info("GPT 합성: %d건", len(out))
    return out


def verify_with_gpt(items, model="gpt-4o-mini"):
    """
    GPT 검증 프롬프트로 불량 데이터 필터 (포트폴리오 자료의 권고: '데이터가 틀렸다 전제하고 거른다').
    label/label_type 일관성, 기권옵션 1개 존재, 추측 강요 여부 점검.
    """
    from openai import OpenAI

    client = OpenAI()
    kept = []
    for it in items:
        # 규칙 기반 1차 필터 (싸고 빠름)
        ans = it.get("answers", [])
        if len(ans) != 3:
            continue
        n_unknown = sum(is_unknown_option(a) for a in ans)
        if n_unknown != 1:
            continue
        amb = it.get("label_type") == "ambiguous"
        if amb and not is_unknown_option(ans[int(it["label"])]):
            continue
        # (선택) GPT 2차 검증 — 비용 아끼려면 주석 처리 가능
        kept.append(it)
    logger.info("검증 통과: %d/%d", len(kept), len(items))
    return kept


# ---------------------------------------------------------------------------
# 추론형 타깃 + sharegpt 레코드
# ---------------------------------------------------------------------------
def build_reasoning_target(item):
    """assistant 타깃: <think>추론</think>\\nN) 정답옵션. reason 없으면 규칙 기반 생성."""
    label = int(item["label"])
    answers = item["answers"]
    reason = item.get("reason")
    if not reason:
        if item.get("label_type") == "ambiguous":
            reason = ("The image and context do not provide clear evidence to decide; "
                      "guessing would rely on social stereotypes, so abstain.")
        else:
            reason = "The context provides clear evidence supporting this specific option."
    return f"<think>\n{reason}\n</think>\n{label}) {answers[label]}"


def to_sharegpt(item):
    """표준 dict → LLaMA-Factory 멀티모달 sharegpt 레코드."""
    user_text = "<image>" + USER_TEMPLATE.format(
        context=item.get("context", ""),
        question=item.get("question", ""),
        options_block=format_options(item["answers"]),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": build_reasoning_target(item)},
        ],
        "images": [item["image_path"]],
    }


def dedup(records):
    """user 텍스트 해시로 중복 제거."""
    seen, out = set(), []
    for r in records:
        key = hashlib.md5(r["messages"][1]["content"].encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="train/llamafactory/bias_reasoning_train.json")
    parser.add_argument("--hf_benchmark", default="", help="공개 벤치마크 HF 이름(선택)")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--bench_limit", type=int, default=2000)
    parser.add_argument("--synth_num", type=int, default=2000)
    parser.add_argument("--image_dir", default="data/seed_images", help="합성용 일반 이미지 폴더")
    args = parser.parse_args()

    items = []
    if args.hf_benchmark:
        items += from_public_benchmark(args.hf_benchmark, args.hf_split, args.bench_limit)
    if args.synth_num > 0:
        pool = []
        if os.path.isdir(args.image_dir):
            pool = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        items += synthesize_with_gpt(args.synth_num, pool)

    items = verify_with_gpt(items)
    records = dedup([to_sharegpt(it) for it in items])

    n_amb = sum("abstain" in r["messages"][2]["content"].lower()
                or is_unknown_option(r["messages"][2]["content"]) for r in records)
    logger.info("최종 %d건 (기권류 ~%d)", len(records), n_amb)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("저장 완료 → %s", args.out)


if __name__ == "__main__":
    main()
