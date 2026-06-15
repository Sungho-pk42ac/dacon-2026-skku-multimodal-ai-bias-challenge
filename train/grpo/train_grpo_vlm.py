"""
VLM GRPO (RL) — v2(SFT) 위에 강화학습. TRL 0.19 GRPOTrainer + Qwen2.5-VL.
보상 = 생성 답이 정답 인덱스와 일치하면 +1 (기권 정답 포함). 별도 grpo_env에서 실행.

전략: BBQ에서 prompt(이미지+context+질문+선택지) → 모델이 생성 → text_to_label로 인덱스 추출 → gold와 비교.
※ 별도 venv(/workspace/grpo_env): trl0.19/transformers4.52/torch2.6.
실행: python train/grpo/train_grpo_vlm.py --base outputs/merged_v2 --n 1200 --steps 300
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label, UNKNOWN_PATTERNS

SEED = 42
random.seed(SEED)


def is_unknown(opt):
    o = str(opt).lower()
    return any(p in o for p in UNKNOWN_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v2", help="GRPO 시작점(SFT merged)")
    ap.add_argument("--out", default="outputs/grpo_v4")
    ap.add_argument("--hf", default="walledai/BBQ")
    ap.add_argument("--n", type=int, default=1200, help="GRPO 프롬프트 수(균형)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--num_gen", type=int, default=4)
    args = ap.parse_args()

    import torch
    from datasets import Dataset, load_dataset
    from PIL import Image
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    # placeholder 이미지(SFT와 동일 정책)
    img_path = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img_path):
        os.makedirs("data", exist_ok=True)
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img_path)
    placeholder = Image.open(img_path).convert("RGB")

    # BBQ → 균형 표본 → GRPO 데이터셋(prompt 대화형 + image + gold_label + answers)
    d = load_dataset(args.hf)
    pool = []
    for split in d.keys():
        for r in d[split]:
            ans = list(r["choices"])
            if len(ans) != 3:
                continue
            lab = int(r["answer"])
            pool.append({"context": r["context"], "question": r["question"], "answers": ans,
                         "label": lab, "amb": is_unknown(ans[lab])})
    amb = [x for x in pool if x["amb"]]
    dis = [x for x in pool if not x["amb"]]
    random.shuffle(amb); random.shuffle(dis)
    half = args.n // 2
    items = amb[:half] + dis[:half]
    random.shuffle(items)

    def to_row(it):
        user_text = USER_TEMPLATE.format(context=it["context"], question=it["question"],
                                         options_block=format_options(it["answers"]))
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            ],
            "image": placeholder,
            "gold_label": it["label"],
            "answers_json": str(it["answers"]),
        }

    ds = Dataset.from_list([to_row(it) for it in items])

    # 보상: 생성 답 → 인덱스 → gold 일치시 +1
    def reward_correct(completions, gold_label, answers_json, **kwargs):
        import ast
        out = []
        for comp, gold, aj in zip(completions, gold_label, answers_json):
            text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
            try:
                answers = ast.literal_eval(aj)
            except Exception:
                answers = ["", "", ""]
            pred = text_to_label(text, answers)
            out.append(1.0 if pred == int(gold) else 0.0)
        return out

    config = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.num_gen,   # 한 프롬프트의 그룹을 한 배치에
        num_generations=args.num_gen,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        max_prompt_length=1024,
        max_completion_length=200,
        max_steps=args.steps,
        logging_steps=5,
        save_steps=args.steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="grpo_v4",
        temperature=0.9,
        beta=0.04,
    )

    # LoRA로 GRPO (풀FT는 48GB OOM). merged_v2 위에 어댑터 학습 → 후처리 merge.
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    trainer = GRPOTrainer(
        model=args.base,
        reward_funcs=reward_correct,
        args=config,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)  # LoRA 어댑터 저장 (base=merged_v2)
    print("GRPO_DONE", args.out)


if __name__ == "__main__":
    main()
