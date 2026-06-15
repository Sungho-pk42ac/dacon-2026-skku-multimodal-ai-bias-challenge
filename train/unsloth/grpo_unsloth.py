"""
Unsloth GRPO (v5) — FastVisionModel Qwen3-VL + vLLM fast_inference. 회원님 train_grpo.ipynb 기반.

목적(스펙): 긴 reasoning 유도 X. 빠른 단일토큰(0/1/2) 선택 정책을 RL로 강화.
일관성 보장: make_bbq_clean의 sample_id/shuffle_opts/is_unknown을 import → v4와 동일 셔플·누출배제.
reward: accuracy(정답일치) + format(0/1/2 단일토큰) — 회원님 reward 함수 리스트 패턴.

실행: python train/unsloth/grpo_unsloth.py --base outputs/merged_v4 --out outputs/merged_v5 \
        --val_ids data/val_ids.json --n 2000 --steps 300 --rank 32 --num_gen 4
"""

import argparse
import ast
import json
import os
import random
import sys

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")  # 템플릿: 속도저하 최소화

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
from data_build.make_bbq_clean import sample_id, shuffle_opts, is_unknown  # 일관성: v4와 동일 함수

SEED = 42
random.seed(SEED)


def _completion_text(comp):
    if isinstance(comp, list) and comp:
        c = comp[-1].get("content", "") if isinstance(comp[-1], dict) else comp[-1]
        return c if isinstance(c, str) else str(c)
    return str(comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--out", default="outputs/merged_v5")
    ap.add_argument("--val_ids", default="data/val_ids.json")
    ap.add_argument("--hf", default="walledai/BBQ")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num_gen", type=int, default=4)
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset, load_dataset
    from PIL import Image

    # 1) 모델 + fast_inference (회원님 템플릿)
    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, fast_inference=True,
        max_lora_rank=args.rank, gpu_memory_utilization=0.85, max_seq_length=1024,
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, use_gradient_checkpointing="unsloth", random_state=3407,
    )

    # 2) 누출 배제 + 옵션셔플(make_bbq_clean과 동일 함수 → 일관성)
    val_ids = set(json.load(open(args.val_ids, encoding="utf-8"))) if os.path.exists(args.val_ids) else set()
    img = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img)
    placeholder = Image.open(img).convert("RGB")

    d = load_dataset(args.hf)
    pool = []
    for split in d.keys():
        for r in d[split]:
            ans = list(r["choices"])
            if len(ans) != 3:
                continue
            sid = sample_id(r["context"], r["question"], ans)
            if sid in val_ids:                       # 누출 배제(SSOT)
                continue
            lab = int(r["answer"])
            na, nl = shuffle_opts(ans, lab, sid)     # 옵션셔플(v4 일관)
            pool.append({"context": r["context"], "question": r["question"],
                         "answers": na, "label": nl, "amb": is_unknown(na[nl])})
    amb = [x for x in pool if x["amb"]]
    dis = [x for x in pool if not x["amb"]]
    random.shuffle(amb); random.shuffle(dis)
    half = args.n // 2
    items = amb[:half] + dis[:half]
    random.shuffle(items)
    print(f"[GRPO] {len(items)} 프롬프트 (val 배제 {len(val_ids)}, 옵션셔플 적용)")

    def to_row(it):
        user = USER_TEMPLATE.format(context=it["context"], question=it["question"],
                                    options_block=format_options(it["answers"]))
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "image", "image": placeholder},
                                             {"type": "text", "text": user}]},
            ],
            "gold": it["label"], "answers": str(it["answers"]),
        }
    ds = Dataset.from_list([to_row(it) for it in items])

    # 3) reward 함수 리스트 (회원님 패턴) — 단일토큰 0/1/2
    def reward_accuracy(completions, gold, answers, **kw):
        out = []
        for comp, g, aj in zip(completions, gold, answers):
            text = _completion_text(comp)
            try:
                A = ast.literal_eval(aj)
            except Exception:
                A = ["", "", ""]
            out.append(2.0 if text_to_label(text, A) == int(g) else -0.5)
        return out

    def reward_format(completions, **kw):
        out = []
        for comp in completions:
            t = _completion_text(comp).strip()
            out.append(0.5 if (0 < len(t) <= 2 and t[:1] in ("0", "1", "2")) else -1.0)
        return out

    cfg = GRPOConfig(
        learning_rate=5e-6, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_8bit", per_device_train_batch_size=args.num_gen, num_generations=args.num_gen,
        gradient_accumulation_steps=4, max_prompt_length=1024, max_completion_length=4,
        max_steps=args.steps, save_steps=args.steps, logging_steps=5, temperature=0.9,
        bf16=True, report_to="wandb", run_name="grpo-v5-unsloth", output_dir="outputs/grpo_v5_unsloth",
    )
    trainer = GRPOTrainer(
        model=model, processing_class=processor,
        reward_funcs=[reward_accuracy, reward_format],
        args=cfg, train_dataset=ds,
    )
    trainer.train()
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print("GRPO_V5_DONE", args.out)


if __name__ == "__main__":
    main()
