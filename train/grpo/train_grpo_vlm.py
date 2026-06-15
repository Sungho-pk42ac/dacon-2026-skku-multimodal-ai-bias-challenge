"""
v5 VLM GRPO (RL) — v4(SFT) 위에 강화학습. trl 1.6 GRPOTrainer + Qwen3-VL.
보상 = 생성 답 인덱스가 정답과 일치하면 +1 (기권 정답 포함) → Balanced Accuracy 직접 최적화.

회의 합의 반영:
  - **누출 제거**: data/val_ids.json(SSOT) 에 든 sample_id는 GRPO 학습에서 전역 배제(정직한 측정 보장).
  - **단일토큰 직답**: max_completion_length 작게(기본 8) → 속도 + "0/1/2" 정책에 정책경사.
  - 보상은 단순(정답 일치, +ambiguous 약가중) — 회의: 5종 보상 과설계 금지.
별도 venv(/workspace/grpo3_env): trl1.6/transformers5.6/torch2.6, Qwen3VL 지원 확인됨.
실행: python train/grpo/train_grpo_vlm.py --base outputs/merged_v4 --out outputs/grpo_v5 --n 1500 --steps 250
"""

import argparse
import hashlib
import json
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


def sample_id(context, question, answers):
    """make_bbq_clean.py와 동일 — val 누출 배제 키."""
    key = "|".join([str(context).strip(), str(question).strip(), "||".join(str(a).strip() for a in answers)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def shuffle_opts(answers, label, seed_str):
    """make_bbq_clean.py와 동일 — 선택지 셔플+라벨 재매핑(위치편향 제거, v4와 일관)."""
    rng = random.Random(seed_str)
    order = [0, 1, 2]
    rng.shuffle(order)
    return [answers[i] for i in order], order.index(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4", help="GRPO 시작점(v4 SFT merged, Qwen3-VL)")
    ap.add_argument("--out", default="outputs/grpo_v5")
    ap.add_argument("--hf", default="walledai/BBQ")
    ap.add_argument("--val_ids", default="data/val_ids.json", help="누출 배제 SSOT")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--num_gen", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=8, help="단일토큰 직답이라 작게")
    args = ap.parse_args()

    import torch
    from datasets import Dataset, load_dataset
    from PIL import Image
    from peft import LoraConfig
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    # 누출 배제 집합
    val_ids = set()
    if os.path.exists(args.val_ids):
        with open(args.val_ids, encoding="utf-8") as f:
            val_ids = set(json.load(f))
    print(f"[GRPO] val_ids 배제 {len(val_ids)}건 로드")

    img_path = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img_path):
        os.makedirs("data", exist_ok=True)
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img_path)
    placeholder = Image.open(img_path).convert("RGB")

    # BBQ → 누출배제 → 균형표본
    d = load_dataset(args.hf)
    pool, skipped = [], 0
    for split in d.keys():
        for r in d[split]:
            ans = list(r["choices"])
            if len(ans) != 3:
                continue
            sid = sample_id(r["context"], r["question"], ans)
            if sid in val_ids:        # 누출 배제
                skipped += 1
                continue
            lab = int(r["answer"])
            pool.append({"context": r["context"], "question": r["question"], "answers": ans,
                         "label": lab, "amb": is_unknown(ans[lab]), "sid": sid})
    print(f"[GRPO] pool {len(pool)} (val 누출 {skipped}건 배제)")

    amb = [x for x in pool if x["amb"]]
    dis = [x for x in pool if not x["amb"]]
    random.shuffle(amb); random.shuffle(dis)
    half = args.n // 2
    items = amb[:half] + dis[:half]
    random.shuffle(items)

    def to_row(it):
        ans, lab = shuffle_opts(it["answers"], it["label"], it["sid"])  # 위치편향 제거(v4와 일관)
        user_text = USER_TEMPLATE.format(context=it["context"], question=it["question"],
                                         options_block=format_options(ans))
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            ],
            "image": placeholder,
            "gold_label": lab,
            "answers_json": str(ans),
            "amb": it["amb"],
        }

    ds = Dataset.from_list([to_row(it) for it in items])

    # 보상(Tier1): accuracy + ambiguous(기권) 가중 + format(0/1/2 단일토큰 준수).
    # 스펙의 10종 reward 중 우리 데이터로 의미있는 것만(나머지=conflict/OCR/cf는 데이터 부재로 제외).
    def reward_correct(completions, gold_label, answers_json, amb, **kwargs):
        import ast
        out = []
        for comp, gold, aj, is_amb in zip(completions, gold_label, answers_json, amb):
            text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
            try:
                answers = ast.literal_eval(aj)
            except Exception:
                answers = ["", "", ""]
            raw = str(text).strip()
            valid_single = (0 < len(raw) <= 2 and raw[:1] in ("0", "1", "2"))  # 단일토큰 형식
            fmt = 0.2 if valid_single else -0.3            # format_reward / invalid_token_penalty
            pred = text_to_label(text, answers)
            acc = (1.2 if is_amb else 1.0) if pred == int(gold) else 0.0  # accuracy + unknown 가중
            out.append(acc + fmt)
        return out

    config = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.num_gen,
        num_generations=args.num_gen,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        max_prompt_length=1024,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        logging_steps=5,
        save_steps=args.steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="grpo_v5",
        temperature=1.0,
        beta=0.04,
    )

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    trainer = GRPOTrainer(
        model=args.base,
        reward_funcs=reward_correct,
        args=config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(args.out)
    print("GRPO_V5_DONE", args.out)


if __name__ == "__main__":
    main()
