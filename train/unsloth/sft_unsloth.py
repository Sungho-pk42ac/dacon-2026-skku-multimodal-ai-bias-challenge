"""
Unsloth SFT (v4) — FastVisionModel Qwen3-VL, 단일토큰 직답. 회원님 02_unsloth 템플릿 기반.

일관성 보장: make_bbq_clean.py가 만든 bbq_v4_train.json(옵션셔플+시나리오dedup+누출제거+단일토큰)
            을 *그대로* 재사용 → LLaMA-Factory 버전과 동일 데이터. val_ids.json은 SSOT로 공유.

실행: python train/unsloth/sft_unsloth.py --model Qwen/Qwen3-VL-8B-Instruct \
        --train data/bbq_v4_train.json --out outputs/merged_v4 --epochs 3 --rank 32
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--train", default="data/bbq_v4_train.json")
    ap.add_argument("--out", default="outputs/merged_v4")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max_steps", type=int, default=0, help=">0이면 epochs 대신 스텝 제한")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--qlora", action="store_true", help="4bit QLoRA(메모리 부족시)")
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from PIL import Image

    # 1) 모델 로드 (Unsloth Qwen3-VL)
    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=args.qlora,
        use_gradient_checkpointing="unsloth",
        max_seq_length=2048,
    )
    # 비전 타워는 동결(placeholder라 학습 불필요) — 언어/어텐션/MLP만 LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
        random_state=3407,
    )
    FastVisionModel.for_training(model)

    # 2) bbq_v4_train.json(sharegpt) → Unsloth vision 대화형 포맷
    raw = json.load(open(args.train, encoding="utf-8"))
    _cache = {}

    def load_img(p):
        if p not in _cache:
            _cache[p] = Image.open(p).convert("RGB")
        return _cache[p]

    def convert(rec):
        img = load_img(rec["images"][0])
        msgs = []
        for m in rec["messages"]:
            if m["role"] == "system":
                msgs.append({"role": "system", "content": [{"type": "text", "text": m["content"]}]})
            elif m["role"] == "user":
                txt = m["content"].replace("<image>", "", 1).lstrip()
                msgs.append({"role": "user", "content": [
                    {"type": "image", "image": img}, {"type": "text", "text": txt}]})
            else:
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": m["content"]}]})
        return {"messages": msgs}

    ds = Dataset.from_list([convert(r) for r in raw])
    print(f"[SFT] 학습 샘플 {len(ds)} (bbq_v4_train.json 재사용 — 일관성 보장)")

    # 3) SFT (TRL + Unsloth vision collator)
    cfg = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        output_dir="outputs/sft_v4_unsloth",
        report_to="wandb",
        run_name="sft-v4-unsloth-qwen3vl",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=ds,
        args=cfg,
    )
    trainer.train()

    # 4) 병합 저장(16bit merged) — 추론/제출용 단일 가중치
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print(f"SFT_DONE merged → {args.out}")


if __name__ == "__main__":
    main()
