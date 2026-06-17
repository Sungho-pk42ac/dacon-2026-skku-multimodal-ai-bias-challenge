"""
v8 Robust-GRPO (Phase 6). 베이스=outputs/merged_v4 (콜드스타트 X, v7 미사용).
코드 참조: grpo_hard_v6.py. 통합 스키마(context/question 분리) 입력.
가설: accuracy-only GRPO가 v4를 못 넘은 건 보상 포화 → 강건성 보상으로 기권/형식/반고정관념 개선 시도.

보상 6종(각 컴포넌트 개별 로깅):
 1 answer        정답+2.0 / 오답-1.0
 2 format        단일토큰 0/1/2 또는 <SOLUTION>idx +0.3 / 위반 -0.5
 3 abstain_consistency  amb&pred==unknown +1.0 / amb&pred!=unknown -1.0 / dis 과기권 -0.5
 4 stereotype_penalty   reasoning이 인구통계 추측을 근거로 쓰면 -0.5 (보수적; v8B만 유효)
 5 length_penalty       길이 과다 페널티 (단일토큰은 짧을수록 선호)
 (option_shuffle_consistency: 온라인 쌍 보상은 복잡 → v8은 풀에 셔플변형 포함(데이터증강)으로 순서불변 학습.
   reward판 셔플일관성은 향후과제로 문서화. 투기적 보상 남발 금지 원칙.)

실행(v8A 단일토큰): python train/unsloth/grpo_robust_v8.py --base outputs/merged_v4 \
   --pool /workspace/v8_pool.json --out outputs/merged_v8a --rank 32 --num_gen 8 --max_completion 4 --mode single
실행(v8B evidence):  ... --max_completion 32 --mode evidence
"""
import argparse, ast, json, os, re, random, sys
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")  # v7 크래시 회피: sleep 토글 비활성
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label, UNKNOWN_PATTERNS
SEED = 42; random.seed(SEED)

R_START, R_END, S_START, S_END = "<start_working_out>", "<end_working_out>", "<SOLUTION>", "</SOLUTION>"
SYS_EVID = (SYSTEM_MESSAGE + f"\nBriefly state evidence between {R_START} and {R_END} (one short clause), "
            f"then the index between {S_START} and {S_END}.")
_solre = re.compile(rf"{S_START}\s*([012])", re.DOTALL)
# 인구통계 추측 근거 사용 탐지(보수적): 명백한 패턴만
_stereo = re.compile(r"\b(because (he|she|men|women|man|woman|the (man|woman|male|female))|"
                     r"(males?|females?|men|women) (are|tend|usually|typically))\b", re.I)


def ctext(c):
    if isinstance(c, list) and c:
        x = c[-1]; return x.get("content", "") if isinstance(x, dict) else str(x)
    if isinstance(c, dict): return str(c.get("content", ""))
    return str(c)


def to_idx(text, answers, mode):
    if mode == "evidence":
        mm = _solre.search(text)
        if mm: return int(mm.group(1))
        text = text.split(S_START)[-1] if S_START in text else text
    return text_to_label(text, answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--pool", default="/workspace/v8_pool.json")
    ap.add_argument("--out", default="outputs/merged_v8a")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max_completion", type=int, default=4)
    ap.add_argument("--mode", choices=["single", "evidence"], default="single")
    ap.add_argument("--finetune_mlp", action="store_true", help="기본 비활성(attention 우선)")
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from PIL import Image

    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, fast_inference=True,
        max_lora_rank=args.rank, gpu_memory_utilization=0.6, max_seq_length=1024)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=args.finetune_mlp,  # MLP 기본 OFF
        r=args.rank, lora_alpha=args.rank, use_gradient_checkpointing="unsloth", random_state=3407)

    img = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img)
    placeholder = Image.open(img).convert("RGB")

    sysmsg = SYS_EVID if args.mode == "evidence" else SYSTEM_MESSAGE
    pool = [r for r in json.load(open(args.pool, encoding="utf-8")) if len(r.get("answers", [])) == 3]
    random.shuffle(pool)
    from collections import Counter
    print(f"[GRPO-v8 {args.mode}] pool {len(pool)} src {dict(Counter(r['src'] for r in pool))}", flush=True)

    def to_row(it):
        user = USER_TEMPLATE.format(context=it.get("context", ""), question=it["question"],
                                    options_block=format_options(it["answers"]))
        return {"prompt": [{"role": "system", "content": [{"type": "text", "text": sysmsg}]},
                           {"role": "user", "content": [{"type": "image", "image": placeholder},
                                                        {"type": "text", "text": user}]}],
                "gold": int(it["label"]), "answers": str(it["answers"]),
                "unk": -1 if it.get("unknown_idx") is None else int(it["unknown_idx"]),
                "src": it["src"]}
    ds = Dataset.from_list([to_row(it) for it in pool])

    # ---- 보상 6종 (각 개별; TRL이 rewards/<함수명>/mean 으로 로깅) ----
    def _A(aj):
        try: return ast.literal_eval(aj)
        except Exception: return ["", "", ""]

    def reward_answer(completions, gold, answers, **kw):
        return [2.0 if to_idx(ctext(c), _A(a), args.mode) == int(g) else -1.0
                for c, g, a in zip(completions, gold, answers)]

    def reward_format(completions, **kw):
        out = []
        for c in completions:
            t = ctext(c).strip()
            if args.mode == "evidence":
                ok = _solre.search(t) is not None
            else:
                ok = 0 < len(t) <= 2 and t[:1] in ("0", "1", "2")
            out.append(0.3 if ok else -0.5)
        return out

    def reward_abstain_consistency(completions, gold, answers, unk, src, **kw):
        out = []
        for c, g, a, u, s in zip(completions, gold, answers, unk, src):
            A = _A(a); p = to_idx(ctext(c), A, args.mode)
            if u < 0: out.append(0.0); continue            # 기권옵션 없는 OOD는 중립
            if str(s) == "bbq_amb":
                out.append(1.0 if p == u else -1.0)
            elif str(s) == "bbq_dis":
                out.append(-0.5 if (p == u and int(g) != u) else 0.0)
            else: out.append(0.0)
        return out

    def reward_stereotype_penalty(completions, **kw):
        # v8B(evidence)만 의미. 단일토큰엔 reasoning 없어 항상 0.
        if args.mode != "evidence": return [0.0] * len(completions)
        return [-0.5 if _stereo.search(ctext(c)) else 0.0 for c in completions]

    def reward_length_penalty(completions, **kw):
        out = []
        for c in completions:
            n = len(ctext(c).strip())
            if args.mode == "single":
                out.append(0.0 if n <= 2 else -0.3)        # 단일토큰: 짧게 강선호
            else:
                out.append(0.0 if n <= 160 else -0.3)      # evidence: 과길이만 페널티
        return out

    cfg = GRPOConfig(
        learning_rate=5e-6, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_8bit", per_device_train_batch_size=args.num_gen, num_generations=args.num_gen,
        gradient_accumulation_steps=4, max_prompt_length=640, max_completion_length=args.max_completion,
        max_steps=args.steps, save_steps=50, logging_steps=5, temperature=0.9, max_grad_norm=1.0,
        bf16=True, report_to="wandb", run_name=f"grpo-v8-{args.mode}", output_dir=f"outputs/grpo_v8_{args.mode}")
    trainer = GRPOTrainer(
        model=model, processing_class=processor,
        reward_funcs=[reward_answer, reward_format, reward_abstain_consistency,
                      reward_stereotype_penalty, reward_length_penalty],
        args=cfg, train_dataset=ds)
    trainer.train()
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print("GRPO_V8_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
