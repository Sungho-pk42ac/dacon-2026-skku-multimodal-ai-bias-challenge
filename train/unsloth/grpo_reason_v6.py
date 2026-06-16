"""
Unsloth GRPO v6 (REASONING) — 회원님 train_grpo.ipynb 베이스 이식. FastVisionModel(Qwen3-VL)+vLLM.

왜 추론인가: 단일토큰 GRPO는 이미 결정적 분류기라 개선 여지가 거의 없음(v6 1차 실패 원인).
베이스코드처럼 "생각 후 답" + 단계적 보상을 주면 GRPO가 *추론 경로*를 최적화 → 정확도/공정성 실제 개선 여지.
편향 과제에서 숙고(reasoning)는 고정관념 단축경로를 차단하는 정석 기법.

형식: <start_working_out>…사고…<end_working_out><SOLUTION>0|1|2</SOLUTION>
보상(베이스 4종 이식→3종): ①형식정확(+3) ②형식근사(태그별 ±) ③정답인덱스(점진적).
학습데이터: hardpool.json(v4 오답 하드네거티브 + BBQ 기권앵커) — 규칙안전 공개 일반/편향 QA.

실행: python train/unsloth/grpo_reason_v6.py --base outputs/merged_v4 --out outputs/merged_v6r \
        --hardpool /workspace/hardpool.json --steps 500 --rank 64 --num_gen 4
"""

import argparse
import json
import os
import random
import re
import sys

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import format_options  # 선택지 블록만 재사용(시스템/유저는 추론용으로 신규)

SEED = 42
random.seed(SEED)

R_START = "<start_working_out>"
R_END = "<end_working_out>"
S_START = "<SOLUTION>"
S_END = "</SOLUTION>"

SYSTEM_REASON = (
    "You are a careful assistant for multiple-choice questions that probe social bias.\n"
    "First reason step by step about the evidence in the context. "
    "Do NOT rely on social stereotypes (gender, race, ethnicity, appearance). "
    "Only commit to a person/option when the context gives clear evidence; "
    "otherwise choose the 'cannot be determined / not enough information' option.\n"
    f"Put your reasoning between {R_START} and {R_END}. Keep the reasoning to ONE short sentence. "
    f"Then output ONLY the index (0, 1, or 2) of the best option between {S_START} and {S_END}."
)
USER_REASON = (
    "Context: {context}\nQuestion: {question}\nOptions:\n{options_block}\n"
    f"Reason briefly, then give the index in {S_START}...{S_END}."
)

# 형식 정규식 — 사고 후 SOLUTION에 0/1/2 단일 인덱스
match_format = re.compile(
    rf"{R_START}.+?{R_END}.*?{S_START}\s*([012])\s*{S_END}",
    flags=re.MULTILINE | re.DOTALL,
)
match_index = re.compile(rf"{S_START}\s*([012])", flags=re.MULTILINE | re.DOTALL)


def _ctext(comp):
    """GRPO completion(list/dict/str) → 텍스트(비전 GRPO 호환)."""
    if isinstance(comp, list) and comp:
        c = comp[-1].get("content", "") if isinstance(comp[-1], dict) else comp[-1]
        return c if isinstance(c, str) else str(c)
    if isinstance(comp, dict):
        return str(comp.get("content", ""))
    return str(comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--out", default="outputs/merged_v6r")
    ap.add_argument("--hardpool", default="/workspace/hardpool.json")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--num_gen", type=int, default=4)
    ap.add_argument("--max_completion", type=int, default=96)
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from PIL import Image

    # 1) 모델 + fast_inference. enforce_eager=True로 CUDA graph illegal-memory 크래시 방지.
    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, fast_inference=True,
        max_lora_rank=args.rank, gpu_memory_utilization=0.7, max_seq_length=1024,
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank, use_gradient_checkpointing="unsloth", random_state=3407,
    )

    img = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img)
    placeholder = Image.open(img).convert("RGB")

    # 2) 하드풀 로드
    pool = json.load(open(args.hardpool, encoding="utf-8"))
    pool = [r for r in pool if isinstance(r.get("o"), list) and len(r["o"]) == 3]
    random.shuffle(pool)
    from collections import Counter
    print(f"[GRPOr-v6] 하드풀 {len(pool)} | 출처 {dict(Counter(r.get('src','?') for r in pool))}", flush=True)

    def to_row(it):
        user = USER_REASON.format(context="", question=it["q"], options_block=format_options(it["o"]))
        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_REASON}]},
                {"role": "user", "content": [{"type": "image", "image": placeholder},
                                             {"type": "text", "text": user}]},
            ],
            "gold": int(it["g"]),
        }
    ds = Dataset.from_list([to_row(it) for it in pool])

    # 3) 보상 함수 (베이스코드 이식, 3지선다 인덱스용)
    def reward_format_exact(completions, **kw):
        return [3.0 if match_format.search(_ctext(c)) is not None else 0.0 for c in completions]

    def reward_format_approx(completions, **kw):
        out = []
        for c in completions:
            r = _ctext(c); s = 0.0
            s += 0.5 if r.count(R_START) == 1 else -0.5
            s += 0.5 if r.count(R_END) == 1 else -0.5
            s += 0.5 if r.count(S_START) == 1 else -0.5
            s += 0.5 if r.count(S_END) == 1 else -0.5
            out.append(s)
        return out

    def reward_answer(completions, gold, **kw):
        out = []
        for c, g in zip(completions, gold):
            r = _ctext(c)
            m = match_format.search(r) or match_index.search(r)
            if m is None:
                out.append(-1.0); continue          # 형식 못맞춤 = 약벌점
            pred = int(m.group(1))
            out.append(3.0 if pred == int(g) else -1.5)  # 정답 강보상 / 오답 벌점(베이스코드 패턴)
        return out

    cfg = GRPOConfig(
        learning_rate=5e-6, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_8bit", per_device_train_batch_size=args.num_gen, num_generations=args.num_gen,
        gradient_accumulation_steps=4, max_prompt_length=512, max_completion_length=args.max_completion,
        max_steps=args.steps, save_steps=args.steps, logging_steps=5, temperature=1.0,
        max_grad_norm=1.0, bf16=True, report_to="wandb",
        run_name="grpo-v6-reasoning", output_dir="outputs/grpo_v6r_unsloth",
    )
    trainer = GRPOTrainer(
        model=model, processing_class=processor,
        reward_funcs=[reward_format_exact, reward_format_approx, reward_answer],
        args=cfg, train_dataset=ds,
    )
    trainer.train()
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print("GRPO_V6R_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
