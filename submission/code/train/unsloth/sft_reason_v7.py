"""
v7 Stage A — SFT 콜드스타트 (DeepSeek-R1 2501.12948 레시피의 1단계).
목적: GRPO 전에 "간결 추론→SOLUTION" 형식을 먼저 안정적으로 학습(형식·정책 동시학습 부담 해소).
데이터: build_v7_data.py가 만든 v7_sft.json [{q,o,g,target}] (규칙기반 추론라벨).
base=merged_v4 → outputs/merged_v7_sft. 이후 grpo_reason_v6로 GRPO(Stage B).

실행: python train/unsloth/sft_reason_v7.py --base outputs/merged_v4 --sft /workspace/v7_sft.json \
        --out outputs/merged_v7_sft --epochs 2 --rank 64
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import format_options

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--sft", default="/workspace/v7_sft.json")
    ap.add_argument("--out", default="outputs/merged_v7_sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--rank", type=int, default=64)
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from PIL import Image

    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, use_gradient_checkpointing="unsloth", max_seq_length=2048)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none", random_state=3407)
    FastVisionModel.for_training(model)

    img = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img)
    placeholder = Image.open(img).convert("RGB")

    raw = json.load(open(args.sft, encoding="utf-8"))

    def conv(rec):
        user = USER_REASON.format(context="", question=rec["q"], options_block=format_options(rec["o"]))
        return {"messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_REASON}]},
            {"role": "user", "content": [{"type": "image", "image": placeholder}, {"type": "text", "text": user}]},
            {"role": "assistant", "content": [{"type": "text", "text": rec["target"]}]},
        ]}
    ds = Dataset.from_list([conv(r) for r in raw])
    print(f"[SFT-v7] 콜드스타트 샘플 {len(ds)}", flush=True)

    cfg = SFTConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_ratio=0.1,
        num_train_epochs=args.epochs, learning_rate=1e-4, lr_scheduler_type="cosine",
        bf16=True, logging_steps=10, save_strategy="epoch", save_total_limit=1,
        optim="adamw_8bit", weight_decay=0.01, seed=42, output_dir="outputs/sft_v7_cold",
        report_to="wandb", run_name="sft-v7-coldstart", remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True}, max_seq_length=2048)
    trainer = SFTTrainer(model=model, tokenizer=processor,
                         data_collator=UnslothVisionDataCollator(model, processor),
                         train_dataset=ds, args=cfg)
    trainer.train()
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print("SFT_V7_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
