"""
Unsloth GRPO (v6) — 하드네거티브 전용 학습 스크립트. base=merged_v4 + vLLM fast_inference.

[v5 대비 개선]
  v5는 BBQ 전체로 GRPO → v4가 이미 잘 맞혀 reward variance≈0 → gradient 소실(무효과).
  v6은 *v4가 틀리는 문제만*(hardpool.json) 학습 → 모든 샘플이 오답 후보 →
  reward 신호가 살아있어 실제로 정책이 개선된다.

[하드풀 출처] (규칙-안전: 모두 독립 공개 일반/편향 QA, 대회 평가셋 비파생)
  - siqa / commonsenseqa / openbookqa / arc 중 v4 오답(3지선다 축소) = 일반추론 하드네거티브
  - BBQ-ambiguous 앵커 = 기권 정책 보존(망각 방지)
  - (선택) UnQover 기권 하드네거티브 = v4가 편향으로 사람을 고른 underspecified 문항

[보상] accuracy(정답 +2.0 / 오답 -0.5) + format(단일토큰 0/1/2 유효 +0.5 / 위반 -1.0). v5와 동일.

[실행]
  python train/unsloth/grpo_hard_v6.py --base outputs/merged_v4 --out outputs/merged_v6 \
      --hardpool /workspace/hardpool.json --steps 400 --rank 32 --num_gen 4
"""

import argparse
import ast
import json
import os
import random
import sys

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")  # 속도저하 최소화(v5 동일)

# prompts.py(저장소 루트)를 import 경로에 추가 — 학습/추론/채굴이 동일 포맷 공유
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

SEED = 42
random.seed(SEED)

# --- 보상값(한 곳에서 관리 — 재현·튜닝 시 여기만 수정. 논문 §3.2 사양) ---
REWARD_ANSWER_CORRECT = 2.0    # 정답 인덱스 일치
REWARD_ANSWER_WRONG = -0.5     # 오답
REWARD_FORMAT_VALID = 0.5      # 단일토큰 0/1/2 형식 준수
REWARD_FORMAT_INVALID = -1.0   # 형식 위반


def parse_args():
    """커맨드라인 인자 파싱."""
    ap = argparse.ArgumentParser(description="v6 하드네거티브 GRPO 학습")
    ap.add_argument("--base", default="outputs/merged_v4", help="베이스 모델(merged_v4)")
    ap.add_argument("--out", default="outputs/merged_v6", help="병합 결과 저장 경로")
    ap.add_argument("--hardpool", default="/workspace/hardpool.json", help="하드네거티브 풀 JSON")
    ap.add_argument("--steps", type=int, default=400, help="GRPO 최대 스텝")
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank(=alpha)")
    ap.add_argument("--num_gen", type=int, default=4, help="프롬프트당 생성 샘플 수(G)")
    ap.add_argument("--max_items", type=int, default=0, help=">0이면 하드풀 상한")
    return ap.parse_args()


def completion_to_text(completion):
    """GRPO completion(리스트/딕트/문자열) → 순수 텍스트로 정규화."""
    if isinstance(completion, list) and completion:
        last = completion[-1]
        content = last.get("content", "") if isinstance(last, dict) else last
        return content if isinstance(content, str) else str(content)
    return str(completion)


def parse_options(answers_json):
    """문자열로 직렬화된 선택지 리스트 → 파이썬 리스트(파싱 실패 시 빈 3선지)."""
    try:
        return ast.literal_eval(answers_json)
    except Exception:
        return ["", "", ""]


def reward_accuracy(completions, gold, answers, **kw):
    """정답 보상: 파싱한 라벨이 정답 인덱스와 일치하면 +2.0, 아니면 -0.5."""
    rewards = []
    for completion, gold_idx, answers_json in zip(completions, gold, answers):
        text = completion_to_text(completion)
        options = parse_options(answers_json)
        hit = text_to_label(text, options) == int(gold_idx)
        rewards.append(REWARD_ANSWER_CORRECT if hit else REWARD_ANSWER_WRONG)
    return rewards


def reward_format(completions, **kw):
    """형식 보상: 출력이 단일토큰 0/1/2(길이≤2)면 +0.5, 위반이면 -1.0."""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        valid = 0 < len(text) <= 2 and text[:1] in ("0", "1", "2")
        rewards.append(REWARD_FORMAT_VALID if valid else REWARD_FORMAT_INVALID)
    return rewards


def load_hard_pool(path, max_items):
    """하드풀 로드 → 3지선다만 필터 → 셔플(seed 42) → (선택)상한 적용."""
    pool = json.load(open(path, encoding="utf-8"))
    pool = [r for r in pool if isinstance(r.get("o"), list) and len(r["o"]) == 3]
    random.shuffle(pool)
    if max_items > 0:
        pool = pool[:max_items]
    return pool


def build_prompt_row(item, placeholder_image):
    """
    하드풀 한 행 → GRPO 프롬프트 행.

    채굴(mine_hard.py)의 pred()와 동일하게 context=""·question=item["q"]로 구성해
    학습/채굴 분포를 정합시킨다. 이미지는 placeholder(대회 포맷이 VLM이라 형식상 필요).
    """
    user_text = USER_TEMPLATE.format(
        context="", question=item["q"], options_block=format_options(item["o"]),
    )
    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": placeholder_image},
                {"type": "text", "text": user_text},
            ]},
        ],
        "gold": int(item["g"]),
        "answers": str(item["o"]),
    }


def main():
    args = parse_args()

    # 무거운 의존성은 여기서 지연 import — unsloth를 trl/datasets보다 먼저 불러
    # 내부 패치 순서를 보장(이 순서를 바꾸면 안 됨).
    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from PIL import Image
    from collections import Counter

    # 1) 모델 + LoRA 어댑터 (vLLM fast_inference, v5 템플릿 동일)
    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, fast_inference=True,
        max_lora_rank=args.rank, gpu_memory_utilization=0.7, max_seq_length=1024,
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank,
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    # 2) placeholder 이미지 (대회 포맷=VLM이지만 본 과제는 텍스트 근거 기반)
    image_path = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(image_path):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(image_path)
    placeholder = Image.open(image_path).convert("RGB")

    # 3) 하드풀 로드 → 프롬프트 데이터셋
    pool = load_hard_pool(args.hardpool, args.max_items)
    src_dist = Counter(r.get("src", "?") for r in pool)
    print(f"[GRPO-v6] 하드풀 {len(pool)} 샘플 | 출처분포 {dict(src_dist)}", flush=True)
    dataset = Dataset.from_list([build_prompt_row(it, placeholder) for it in pool])

    # 4) GRPO 학습 설정 (v5와 동일 하이퍼파라미터)
    cfg = GRPOConfig(
        learning_rate=5e-6, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_8bit", per_device_train_batch_size=args.num_gen, num_generations=args.num_gen,
        gradient_accumulation_steps=4, max_prompt_length=1024, max_completion_length=4,
        max_steps=args.steps, save_steps=args.steps, logging_steps=5, temperature=0.9,
        bf16=True, report_to="wandb", run_name="grpo-v6-hardneg", output_dir="outputs/grpo_v6_unsloth",
    )

    # 5) 학습 → 16bit 병합 저장
    trainer = GRPOTrainer(
        model=model, processing_class=processor,
        reward_funcs=[reward_accuracy, reward_format],
        args=cfg, train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print("GRPO_V6_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
