"""
VLM QLoRA 학습 스크립트 (학습 전용 — 추론 코드와 분리).

폴더 템플릿 `5주차/trl-main/.../vlm tuning - ScienceQA.ipynb`의 SFTTrainer + QLoRA 파이프라인을
대회용(기권 학습 포함)으로 옮긴 것이다. 클라우드 GPU(A6000 48GB급)에서 실행한다.

* 변경 이유: 노트북 → 재현 가능한 스크립트화(시드 고정, config 분리)로 Private Score 재현성 확보.
* 데이터: data_build/로 만든 train_abstention.jsonl (disambiguated + ambiguous 균형).
"""

import argparse
import logging

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path):
    """YAML 설정 로드."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_messages(sample):
    """
    학습 데이터 한 행 → SFTTrainer messages 포맷 (대회 확정 포맷).

    기대 필드: context, question, answers(3개 선택지 리스트/JSON), label(0/1/2), image_path
    정답 타깃 = answers[label] (옵션 텍스트). ambiguous 샘플은 label이 'cannot be determined'류
    옵션을 가리키므로 기권이 라벨에 내장된다.
    """
    import ast
    import json

    from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options  # 학습/추론 공유 포맷

    answers = sample["answers"]
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except (json.JSONDecodeError, TypeError):
            answers = ast.literal_eval(answers)
    target_text = answers[int(sample["label"])]

    user_text = USER_TEMPLATE.format(
        context=sample["context"],
        question=sample["question"],
        options_block=format_options(answers),
    )
    image = Image.open(sample["image_path"]).convert("RGB")
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]
    }


def build_collate_fn(processor):
    """이미지+텍스트 배치 collator (ScienceQA 템플릿과 동일 구조, 이미지 토큰 손실 마스킹)."""

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(ex["messages"], tokenize=False)
            for ex in examples
        ]
        image_inputs = [process_vision_info(ex["messages"])[0] for ex in examples]
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # 이미지 토큰은 손실 계산에서 제외 (Qwen2/2.5-VL 토큰 인덱스)
        if isinstance(processor, Qwen2VLProcessor):
            image_token_ids = [151652, 151653, 151655]
        else:
            image_token_ids = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]
        for tid in image_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 42))
    logger.info("베이스 모델 로드: %s", cfg["model_id"])

    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(cfg["model_id"])

    # 데이터 로드 (jsonl) → messages 변환
    logger.info("학습 데이터 로드: %s", cfg["train_jsonl"])
    raw = load_dataset("json", data_files=cfg["train_jsonl"], split="train")
    train_dataset = [to_messages(s) for s in raw]
    logger.info("학습 샘플 수: %d", len(train_dataset))

    peft_config = LoraConfig(
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        r=cfg.get("lora_r", 256),
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=cfg.get("output_dir", "outputs"),
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 8),
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=cfg.get("lr", 1e-4),
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    processor.tokenizer.padding_side = "right"  # half-precision 학습 안정화
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        data_collator=build_collate_fn(processor),
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    logger.info("학습 시작")
    trainer.train()
    trainer.save_model(sft_args.output_dir)
    logger.info("학습 완료 → %s", sft_args.output_dir)


if __name__ == "__main__":
    main()
