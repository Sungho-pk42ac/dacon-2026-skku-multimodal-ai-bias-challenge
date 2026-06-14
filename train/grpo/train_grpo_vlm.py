"""
GRPO (RL) 단계 — SFT 다음 Tier-3 / 2차 Novelty 부스터.

받으신 `train_grpo.ipynb`는 Unsloth+TRL GRPOTrainer(텍스트)였다. 우리 대회는 VLM이라
TRL `GRPOTrainer`의 멀티모달(VLM) 지원을 사용한다 (Qwen2.5-VL + vLLM colocate).
출처 근거: TRL 공식 문서가 VLM GRPO + vLLM colocate 지원을 명시.

전략: SFT로 학습한 추론형 VLM을 GRPO로 강화.
  보상(reward) = '정답 인덱스 일치' (정답이 기권 옵션인 ambiguous에서 올바른 기권도 자동 보상).
  → "정답 + 올바른 기권"을 직접 최적화 → Balanced Accuracy의 두 항을 함께 끌어올림.

⚠️ 비용·복잡도 큼. 1차 SFT로 점수 확보 후 도전. H100 권장.
⚠️ TRL/모델 버전 호환은 pod에서 확인 후 고정(pip freeze).
"""

import argparse
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_label(text):
    """모델 출력에서 최종 선택 인덱스(0/1/2) 추출. <think> 뒤 'N) ...' 우선."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.search(r"\b([012])\b", text)
    return int(m.group(1)) if m else -1


def reward_correct(completions, gold_label, **kwargs):
    """
    정답 일치 보상. completions: 생성 텍스트 리스트, gold_label: 정답 인덱스 리스트.
    정답이 기권 옵션인 ambiguous 문항도 '일치=보상'으로 올바른 기권을 보상.
    """
    rewards = []
    for comp, gold in zip(completions, gold_label):
        pred = extract_label(comp if isinstance(comp, str) else comp[0]["content"])
        rewards.append(1.0 if pred == int(gold) else 0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", required=True, help="SFT merged 모델 경로(시작점)")
    parser.add_argument("--dataset", required=True, help="GRPO용 데이터(질문+이미지+gold_label)")
    parser.add_argument("--out", default="outputs/grpo_qwen25vl")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    logger.info("GRPO 시작점(SFT 모델): %s", args.sft_model)
    dataset = load_dataset("json", data_files=args.dataset, split="train")

    config = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_generations=8,            # GRPO 그룹 크기
        learning_rate=1e-6,
        bf16=True,
        max_prompt_length=2048,
        max_completion_length=256,    # 추론 토큰 — 길면 느려짐
        use_vllm=True,                # vLLM colocate 가속 (TRL 문서)
        vllm_mode="colocate",
        report_to="wandb",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=args.sft_model,
        reward_funcs=[reward_correct],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.out)
    logger.info("GRPO 완료 → %s", args.out)


if __name__ == "__main__":
    main()
