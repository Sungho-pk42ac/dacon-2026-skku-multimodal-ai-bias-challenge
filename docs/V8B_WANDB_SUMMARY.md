# v8B W&B 요약 (run: v8B_robust_grpo)

- link: https://wandb.ai/psh243360-korea-university/dacon-bias/runs/xm3p2p81
- entity/project: `psh243360-korea-university` / `dacon-bias`

## 학습 설정
```json
{
  "base": "outputs/merged_v4",
  "out": "outputs/v8b_adapter",
  "rank": 16,
  "lr": 1e-06,
  "steps": 200,
  "num_gen": 8,
  "temperature": 0.85,
  "max_completion": 4,
  "finetune": "attention-only (vision frozen, MLP off)",
  "reward_weights": {
    "answer": "+2/-1",
    "shuffle_consistency": "+1/-1",
    "abstain": "+0.5/-0.5/-0.6",
    "source_normalized": "correct - v8a_src_base",
    "format": "+0.2/-0.5",
    "length": "0/-0.3"
  }
}
```
- wall_clock: 1710.4s (~28분)
- dynamic sampling(채굴 G=8): useful 176 / easy 1295 / hard 89

## 보상 컴포넌트 (W&B 그래프: rewards/<함수>/mean, loss, reward 추이 200스텝)
TRL 로깅 키: `rewards/reward_answer/mean`, `rewards/reward_shuffle_consistency/mean`,
`rewards/reward_abstain/mean`, `rewards/reward_source_normalized/mean`,
`rewards/reward_format/mean`, `rewards/reward_length_penalty/mean`, `reward`(total), `loss`.

## per-source 보상 통계 (answer reward, 범위 -1~+2)
| 소스 | reward_mean | reward_std | n |
|---|---:|---:|---:|
| ood_obqa | 0.2890 | 1.4851 | 2104 |
| ood_csqa | 0.2662 | 1.4817 | 1848 |
| ood_arc | 0.1397 | 1.4561 | 408 |
| ood_siqa | -0.0588 | 1.3920 | 2040 |

- 해석: std ~1.4-1.5의 **건강한 분산**(정답 +2/오답 -1 혼재) → 실제 학습 신호 존재.
  v8A의 죽은 format 신호(-0.5 고정) 대비 개선. siqa가 가장 어려움(평균 음수).
- 풀이 전부 OOD라 abstain/source_normalized(bbq)는 비활성(중립) — OOD 강건성 집중에 적합.

## 로컬 요약 파일
- `outputs/wandb_summary_v8b.csv` (step별 보상 컴포넌트, pod 생성)
- `outputs/v8b_train_log.json` (trainer log_history 전체, pod 생성; outputs/는 gitignore)
