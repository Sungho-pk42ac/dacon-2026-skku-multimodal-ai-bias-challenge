"""
v8B W&B 로컬 요약 생성 (v8B Step 3). outputs/v8b_train_log.json(trainer log_history)에서
보상 컴포넌트 추이 + per-source 보상통계 + 설정을 정리.

산출: outputs/wandb_summary_v8b.csv (step별 reward 컴포넌트),
      docs/V8B_WANDB_SUMMARY.md (설정 + 컴포넌트 처음/끝 + per-source 통계 + 동적카운트)

실행: python inference/gen_wandb_summary_v8b.py
"""
import json, os

R = "/workspace/dacon-bias-challenge"
LOG = R + "/outputs/v8b_train_log.json"
CSV = R + "/outputs/wandb_summary_v8b.csv"
MD = R + "/docs/V8B_WANDB_SUMMARY.md"

# TRL 보상 로깅 키(rewards/<함수명>/mean) → 표시명
REWARD_KEYS = [
    ("rewards/reward_answer/mean", "answer"),
    ("rewards/reward_shuffle_consistency/mean", "shuffle_consistency"),
    ("rewards/reward_abstain/mean", "abstain"),
    ("rewards/reward_source_normalized/mean", "source_normalized"),
    ("rewards/reward_format/mean", "format"),
    ("rewards/reward_length_penalty/mean", "length_penalty"),
    ("reward", "total"),
    ("loss", "loss"),
]


def main():
    if not os.path.exists(LOG):
        print("ERROR: no train log", flush=True); return
    d = json.load(open(LOG, encoding="utf-8"))
    hist = d.get("log_history", [])
    cfg = d.get("config", {})
    src_stats = d.get("src_reward_stats", {})
    dyn = d.get("dynamic", {})

    rows = [h for h in hist if any(k in h for k, _ in REWARD_KEYS)]
    # CSV
    with open(CSV, "w", encoding="utf-8") as f:
        header = ["step"] + [disp for _, disp in REWARD_KEYS]
        f.write(",".join(header) + "\n")
        for h in rows:
            line = [str(h.get("step", ""))] + [str(h.get(k, "")) for k, _ in REWARD_KEYS]
            f.write(",".join(line) + "\n")

    def first_last(key):
        vals = [(h.get("step"), h.get(key)) for h in rows if key in h]
        if not vals:
            return ("-", "-")
        return (vals[0][1], vals[-1][1])

    with open(MD, "w", encoding="utf-8") as f:
        f.write("# v8B W&B 요약 (run: v8B_robust_grpo)\n\n")
        f.write("## 학습 설정\n\n```\n" + json.dumps(cfg, ensure_ascii=False, indent=2) + "\n```\n\n")
        f.write(f"- trainable_params: {d.get('trainable_params','-')} · wall_clock_sec: {d.get('wall_clock_sec','-')}\n")
        f.write(f"- dynamic: useful={dyn.get('useful_count','-')} easy={dyn.get('easy_count','-')} "
                f"hard={dyn.get('hard_count','-')} (G={dyn.get('G','-')})\n\n")
        f.write("## 보상 컴포넌트 추이(처음 → 끝)\n\n| 컴포넌트 | 처음 | 끝 |\n|---|---:|---:|\n")
        for k, disp in REWARD_KEYS:
            a, b = first_last(k)
            f.write(f"| {disp} | {a} | {b} |\n")
        f.write("\n## per-source 보상 통계 (answer reward)\n\n| 소스 | mean | std | n |\n|---|---:|---:|---:|\n")
        for s, st in sorted(src_stats.items()):
            f.write(f"| {s} | {st.get('reward_mean','-')} | {st.get('reward_std','-')} | {st.get('n','-')} |\n")

    print("WANDB_SUMMARY_V8B_DONE", flush=True)


if __name__ == "__main__":
    main()
