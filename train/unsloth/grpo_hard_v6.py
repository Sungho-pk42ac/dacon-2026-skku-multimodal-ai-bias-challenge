"""
Unsloth GRPO (v6) — 하드네거티브 전용. base=merged_v4 + vLLM fast_inference.

핵심(v5 대비 개선): v5는 BBQ 전체로 GRPO → v4가 이미 잘 맞혀 reward variance≈0 → gradient 소실(무효과).
v6은 *v4가 틀리는 문제만*(hardpool.json) 학습 → 모든 샘플이 오답 후보 → reward 신호가 살아있어 실제로 정책이 개선됨.

하드풀 출처(규칙-안전: 모두 독립 공개 일반/편향 QA, 대회 평가셋 비파생):
  - siqa/commonsenseqa/openbookqa/arc 중 v4 오답 (3지선다 축소) = 일반추론 하드네거티브
  - BBQ-ambiguous 앵커 = 기권 정책 보존(망각 방지)
  - (선택) UnQover 기권 하드네거티브 = v4가 편향으로 사람을 고른 underspecified 문항

reward: accuracy(정답일치 +2.0/-0.5) + format(0/1/2 단일토큰 +0.5/-1.0) — v5와 동일.

실행: python train/unsloth/grpo_hard_v6.py --base outputs/merged_v4 --out outputs/merged_v6 \
        --hardpool /workspace/hardpool.json --steps 400 --rank 32 --num_gen 4
"""

import argparse
import ast
import json
import os
import random
import sys

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")  # 속도저하 최소화(v5 동일)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label

SEED = 42
random.seed(SEED)


def _completion_text(comp):
    """GRPO completion(리스트/딕트/문자열) → 순수 텍스트."""
    if isinstance(comp, list) and comp:
        c = comp[-1].get("content", "") if isinstance(comp[-1], dict) else comp[-1]
        return c if isinstance(c, str) else str(c)
    return str(comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--out", default="outputs/merged_v6")
    ap.add_argument("--hardpool", default="/workspace/hardpool.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num_gen", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=0, help=">0이면 하드풀 상한")
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from PIL import Image

    # 1) 모델 + fast_inference (v5 템플릿 동일)
    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, fast_inference=True,
        max_lora_rank=args.rank, gpu_memory_utilization=0.7, max_seq_length=1024,
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, use_gradient_checkpointing="unsloth", random_state=3407,
    )

    # 2) placeholder 이미지 (대회 포맷 = VLM이지만 이미지는 placeholder)
    img = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img)
    placeholder = Image.open(img).convert("RGB")

    # 3) 하드풀 로드 — {"src","q","o":[3],"g":int}. 채굴(mine_hard.py)과 동일 포맷으로 프롬프트 구성.
    pool = json.load(open(args.hardpool, encoding="utf-8"))
    pool = [r for r in pool if isinstance(r.get("o"), list) and len(r["o"]) == 3]
    random.shuffle(pool)
    if args.max_items > 0:
        pool = pool[:args.max_items]
    from collections import Counter
    src_dist = Counter(r.get("src", "?") for r in pool)
    print(f"[GRPO-v6] 하드풀 {len(pool)} 샘플 | 출처분포 {dict(src_dist)}", flush=True)

    def to_row(it):
        # 채굴 시 pred()와 동일하게 context="" + question=it["q"] (분포 일치 → 학습/채굴 정합)
        user = USER_TEMPLATE.format(context="", question=it["q"],
                                    options_block=format_options(it["o"]))
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "image", "image": placeholder},
                                             {"type": "text", "text": user}]},
            ],
            "gold": int(it["g"]), "answers": str(it["o"]),
        }
    ds = Dataset.from_list([to_row(it) for it in pool])

    # 4) reward 함수 (v5와 동일) — 단일토큰 0/1/2
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
        bf16=True, report_to="wandb", run_name="grpo-v6-hardneg", output_dir="outputs/grpo_v6_unsloth",
    )
    trainer = GRPOTrainer(
        model=model, processing_class=processor,
        reward_funcs=[reward_accuracy, reward_format],
        args=cfg, train_dataset=ds,
    )
    trainer.train()
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print("GRPO_V6_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
