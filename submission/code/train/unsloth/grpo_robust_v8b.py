"""
v8B Robust-GRPO (v8B Step 2). 베이스=outputs/merged_v4 (콜드스타트 X, 추론확장 X, v7 미사용).
목표: v8A의 OOD 이득(0.8117)은 보존/개선하면서 옵션순서 강건성(셔플 일관성)을 회복.
설계: 더 낮은 lr(1e-6)·낮은 rank(16)·attention-only로 v4를 덜 흔들고,
      셔플짝 데이터 + 셔플일관성 보상으로 순서불변을 직접 학습.

보상 6종(각 컴포넌트 개별 로깅 — TRL: rewards/<함수명>/mean):
 1 reward_answer              정답 인덱스 일치 +2.0 / 오답 -1.0
 2 reward_shuffle_consistency 예측 옵션 '텍스트'가 정답텍스트(gold_text, 순서무관)와 일치 +1.0 / 불일치 -1.0
                              → 셔플짝(원본+셔플) 풀과 결합 시 순서불변을 직접 최적화. 별도 로깅.
 3 reward_abstain            amb&pred==unk +0.5 / amb&pred!=unk -0.5 / dis 과기권(오답) -0.6 / else 0 (중도)
 4 reward_source_normalized  (정답?1:0) - v8a기준정확도[src]  → 소스별 zero-centered. 하드 소스 상향가중. 별도 로깅.
 5 reward_format             단일토큰 0/1/2 유효 +0.2 / 무효 -0.5  (v8A의 죽은신호 버그 수정: strip 후 첫 글자 검사)
 6 reward_length_penalty     단일토큰 길이 과다 페널티

dynamic sampling: 채굴(build_v8b_pool)에서 v8a G회로 k집계해 useful(0<k<G) 위주 오프라인 근사.
                  easy/hard/useful 카운트는 pool_stats에서 읽어 W&B에 1회 로깅.
출력: LoRA 어댑터만 저장(디스크 절약) → 별도 merge 단계에서 merged_v8b 생성.
규칙: 단일토큰 0/1/2만, 최종답은 모델 생성텍스트에서 파싱. DACON 평가셋 미사용. 외부API 0.

실행: python train/unsloth/grpo_robust_v8b.py --base outputs/merged_v4 \
        --pool /workspace/v8b_pool.json --stats /workspace/v8b_pool_stats.json \
        --out outputs/v8b_adapter --rank 16 --lr 1e-6 --steps 200 --num_gen 8 --temperature 0.85
"""
import argparse, ast, json, os, re, random, sys
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")   # v7 크래시(STANDBY sleep→empty_cache) 회피
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
from prompts import SYSTEM_MESSAGE, USER_TEMPLATE, format_options, text_to_label
SEED = 42; random.seed(SEED)

# 소스별 answer 보상 누적(per-source reward mean/std 로깅용) — 모듈 전역
SRC_REWARDS = defaultdict(list)


def _norm(s):
    return "".join(str(s).split()).strip("'\".").lower()


def ctext(c):
    if isinstance(c, list) and c:
        x = c[-1]; return x.get("content", "") if isinstance(x, dict) else str(x)
    if isinstance(c, dict):
        return str(c.get("content", ""))
    return str(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/merged_v4")
    ap.add_argument("--pool", default="/workspace/v8b_pool.json")
    ap.add_argument("--stats", default="/workspace/v8b_pool_stats.json")
    ap.add_argument("--out", default="outputs/v8b_adapter")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max_completion", type=int, default=4)
    args = ap.parse_args()

    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback
    from datasets import Dataset
    from PIL import Image

    # v8a 소스 기준정확도(source_normalized 기준선). 없으면 0.5.
    src_base = {}
    dyn = {}
    if os.path.exists(args.stats):
        st = json.load(open(args.stats, encoding="utf-8"))
        src_base = st.get("src_base_acc_v8a", {})
        dyn = st.get("dynamic", {})

    model, processor = FastVisionModel.from_pretrained(
        args.base, load_in_4bit=False, fast_inference=True,
        max_lora_rank=args.rank, gpu_memory_utilization=0.6, max_seq_length=1024)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=False,   # attention-only (MLP OFF)
        r=args.rank, lora_alpha=args.rank, use_gradient_checkpointing="unsloth", random_state=3407)

    img = os.path.abspath("data/placeholder.jpg")
    if not os.path.exists(img):
        Image.new("RGB", (336, 336), (127, 127, 127)).save(img)
    placeholder = Image.open(img).convert("RGB")   # 공개 텍스트데이터 학습용 placeholder(규칙상 허용)

    pool = [r for r in json.load(open(args.pool, encoding="utf-8")) if len(r.get("answers", [])) == 3]
    random.shuffle(pool)
    from collections import Counter
    print(f"[GRPO-v8B] pool {len(pool)} src {dict(Counter(r['src'] for r in pool))} "
          f"variant {dict(Counter(r.get('variant','?') for r in pool))}", flush=True)

    def to_row(it):
        user = USER_TEMPLATE.format(context=it.get("context", ""), question=it["question"],
                                    options_block=format_options(it["answers"]))
        gold_text = it.get("gold_text", it["answers"][int(it["label"])])
        return {"prompt": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                           {"role": "user", "content": [{"type": "image", "image": placeholder},
                                                        {"type": "text", "text": user}]}],
                "gold": int(it["label"]), "answers": str(it["answers"]), "gold_text": str(gold_text),
                "unk": -1 if it.get("unknown_idx") is None else int(it["unknown_idx"]),
                "src": it["src"]}
    ds = Dataset.from_list([to_row(it) for it in pool])

    def _A(aj):
        try: return ast.literal_eval(aj)
        except Exception: return ["", "", ""]

    # ---- 보상 6종 ----
    def reward_answer(completions, gold, answers, src, **kw):
        out = []
        for c, g, a, s in zip(completions, gold, answers, src):
            r = 2.0 if text_to_label(ctext(c), _A(a)) == int(g) else -1.0
            SRC_REWARDS[str(s)].append(r)      # per-source 로깅용 누적
            out.append(r)
        return out

    def reward_shuffle_consistency(completions, answers, gold_text, **kw):
        # 예측 옵션의 '텍스트'가 정답텍스트(순서무관)와 일치하는가 → 셔플짝과 결합 시 순서불변 학습
        out = []
        for c, a, gt in zip(completions, answers, gold_text):
            A = _A(a); p = text_to_label(ctext(c), A)
            pred_text = A[p] if 0 <= p < len(A) else ""
            out.append(1.0 if _norm(pred_text) == _norm(gt) else -1.0)
        return out

    def reward_abstain(completions, gold, answers, unk, src, **kw):
        out = []
        for c, g, a, u, s in zip(completions, gold, answers, unk, src):
            p = text_to_label(ctext(c), _A(a))
            if u < 0:
                out.append(0.0); continue           # 기권옵션 없는 OOD는 중립
            if str(s) == "bbq_amb":
                out.append(0.5 if p == u else -0.5)
            elif str(s) == "bbq_dis":
                out.append(-0.6 if (p == u and int(g) != u) else 0.0)
            else:
                out.append(0.0)
        return out

    def reward_source_normalized(completions, gold, answers, src, **kw):
        # (정답?1:0) - v8a기준정확도[src] → 소스별 zero-centered. 하드 소스 beat 시 +.
        out = []
        for c, g, a, s in zip(completions, gold, answers, src):
            correct = 1.0 if text_to_label(ctext(c), _A(a)) == int(g) else 0.0
            out.append(correct - float(src_base.get(str(s), 0.5)))
        return out

    def reward_format(completions, **kw):
        # v8A 버그 수정: strip 후 첫 글자가 0/1/2이고 stripped 길이<=2면 유효
        out = []
        for c in completions:
            t = ctext(c).strip()
            ok = len(t) > 0 and t[0] in ("0", "1", "2") and len(t) <= 2
            out.append(0.2 if ok else -0.5)
        return out

    def reward_length_penalty(completions, **kw):
        return [0.0 if len(ctext(c).strip()) <= 2 else -0.3 for c in completions]

    REWARD_WEIGHTS = {"answer": "+2/-1", "shuffle_consistency": "+1/-1", "abstain": "+0.5/-0.5/-0.6",
                      "source_normalized": "correct-srcbase", "format": "+0.2/-0.5", "length": "0/-0.3"}

    cfg = GRPOConfig(
        learning_rate=args.lr, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_8bit", per_device_train_batch_size=args.num_gen, num_generations=args.num_gen,
        gradient_accumulation_steps=4, max_prompt_length=640, max_completion_length=args.max_completion,
        max_steps=args.steps, save_steps=50, logging_steps=5, temperature=args.temperature, max_grad_norm=1.0,
        bf16=True, report_to="wandb", run_name="v8B_robust_grpo", output_dir="outputs/grpo_v8b")

    # W&B: 설정/동적카운트 1회 로깅 + 학습종료 시 per-source 보상통계
    class V8BLogCallback(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            try:
                import wandb
                if wandb.run is not None:
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    wandb.config.update({
                        "base_checkpoint": args.base, "output_checkpoint": args.out,
                        "lora_rank": args.rank, "learning_rate": args.lr, "steps": args.steps,
                        "num_generations": args.num_gen, "temperature": args.temperature,
                        "reward_weights": REWARD_WEIGHTS, "trainable_params": int(trainable),
                        "source_distribution": dict(Counter(r["src"] for r in pool)),
                        "variant_distribution": dict(Counter(r.get("variant", "?") for r in pool)),
                    }, allow_val_change=True)
                    wandb.log({"dynamic/easy_count": dyn.get("easy_count", 0),
                               "dynamic/hard_count": dyn.get("hard_count", 0),
                               "dynamic/useful_count": dyn.get("useful_count", 0)})
            except Exception as e:
                print("[wandb cfg skip]", str(e)[:80], flush=True)

    trainer = GRPOTrainer(
        model=model, processing_class=processor,
        reward_funcs=[reward_answer, reward_shuffle_consistency, reward_abstain,
                      reward_source_normalized, reward_format, reward_length_penalty],
        args=cfg, train_dataset=ds, callbacks=[V8BLogCallback()])
    import time
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0

    # ---- LoRA 어댑터만 저장(디스크 절약; merged_v8b는 별도 merge 단계) ----
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)

    # ---- per-source 보상통계 + 학습로그 요약 저장(W&B 요약 산출물) ----
    import statistics as stt
    src_stats = {}
    for s, vals in SRC_REWARDS.items():
        if vals:
            src_stats[s] = {"reward_mean": round(stt.mean(vals), 4),
                            "reward_std": round(stt.pstdev(vals), 4), "n": len(vals)}
    log_hist = trainer.state.log_history
    os.makedirs("outputs", exist_ok=True)
    json.dump({"wall_clock_sec": round(wall, 1), "trainable_params":
               int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
               "src_reward_stats": src_stats, "log_history": log_hist,
               "config": {"base": args.base, "out": args.out, "rank": args.rank, "lr": args.lr,
                          "steps": args.steps, "num_gen": args.num_gen, "temperature": args.temperature,
                          "reward_weights": REWARD_WEIGHTS},
               "dynamic": dyn},
              open("outputs/v8b_train_log.json", "w"), ensure_ascii=False, indent=2)
    try:
        import wandb
        if wandb.run is not None:
            for s, d in src_stats.items():
                wandb.run.summary[f"source/{s}/reward_mean"] = d["reward_mean"]
                wandb.run.summary[f"source/{s}/reward_std"] = d["reward_std"]
            wandb.run.summary["train/wall_clock_sec"] = round(wall, 1)
    except Exception:
        pass

    print("SRC_REWARD_STATS", json.dumps(src_stats, ensure_ascii=False), flush=True)
    print(f"GRPO_V8B_DONE adapter={args.out} wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
