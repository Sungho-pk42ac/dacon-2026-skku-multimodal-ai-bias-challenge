# 실험 인덱스 (Experiments) — v4 → v8b

> DACON 2026 성균관대 멀티모달 AI Bias 챌린지(236722) · 팀 **pk42ac** · 단독 참가
> base **Qwen3-VL-8B-Instruct** (Apache-2.0) · 최종 제출 **v6 HardNeg-GRPO** · 공식 채점 경로 **실이미지 0.9628**

이 디렉터리는 **실험(버전)별로** 코드·데이터·학습 명령·결과·판정을 한 장씩 묶은 인덱스입니다.
물리적 소스는 기능별 디렉터리(`data_build/` · `train/unsloth/` · `inference/`)에 그대로 두고,
여기서 **버전 → 정확한 파일/명령/산출물**을 매핑합니다. (재현 진입점: [`submission/실행방법.md`](../submission/실행방법.md))

---

## 계보 (Lineage)

```
v1–v3  Qwen2.5-VL 탐색      (누출 과대평가 / 속도 폐기)   — 아카이브
   │
   ▼
v4     단일토큰 SFT          기준선·모든 RL의 base merged
   │
   ├─ v5  Sat-GRPO (BBQ 포화)        효과 0 (보상 분산≈0)        ✗ 음성
   ├─ v6  HardNeg-GRPO ★             일반화 보존·편향 0          ✓ 최종 제출
   ├─ v7  ColdStart SFT + 추론 GRPO  OOD 붕괴·암기              ✗ 음성(FAIL)
   ├─ v8a Robust-GRPO (rank32)       OOD↑ ↔ 셔플↓ trade-off     ✗ 음성
   └─ v8b Robust-GRPO (rank16)       trade-off 해소·무이득       ✗ 음성(PASS이나 무이득)
```

핵심 서사: **콜드스타트 SFT는 일반화의 적, GRPO-only는 일반화를 해치지 않는다.**
복잡도(포화·콜드스타트·강건성 변형)는 8B에서 단순 HardNeg-GRPO를 유의하게 넘지 못했다 → 동급이면 가장 단순한 v6(Occam).

---

## 표 A — 최종 선택 근거 (외부검증 eval-only · 발표/제출 기준)

출처 [`docs/FINAL_MODEL_SELECTION.md`](../docs/FINAL_MODEL_SELECTION.md). 6개 결정축 중 **v6가 5개 best/tied**.

| 결정축 | base(no-FT) | **v6 ★** | v8a | v8b |
|---|---|---|---|---|
| 편향 balanced | 0.85 | **1.0** | 1.0 | 1.0 |
| 편향 s_AMB (↓) | 0.0867 | **0.0** | 0.0 | 0.0 |
| 인분포 OOD | 0.8467 | **0.855** | 0.8517 | 0.8517 |
| 독립 ood2(미관측) | 0.705 | **0.7783** | (비후보) | 0.775 |
| 셔플 일관성 | 0.9122 | **0.9544** | 0.9544 | 0.9522 |
| image-invariance | 0.8967 | **0.975** | — | 0.9717 |
| 속도 s/샘플 | 0.1074 | 0.0778 | 0.0764 | 0.0767 |

**DACON 실채점(test 8,500):** v6 실이미지 **0.9628**(공식 제출) · v6 텍스트집중 1.0(ablation) · v8b 실이미지 0.9643.

## 표 B — 전체 계보 (held-out `v8_eval` 900, 초기 측정 패스)

출처 [`docs/V4_V8B_COMPLETE_COMPARISON.md`](../docs/V4_V8B_COMPLETE_COMPARISON.md).

| 실험 | 학습 | balanced(amb) | OOD | 셔플 일관성 | 판정 | 카드 |
|---|---|---|---|---|---|---|
| **v4** SFT | 단일토큰 SFT | 1.0 | 0.8033 | 0.9478 | 기준선 | [v4-sft](v4-sft.md) |
| **v5** Sat-GRPO | +GRPO·BBQ 포화 | ≈1.0 | ≈0.803 | ≈0.948 | ✗ 효과 0 | [v5-sat-grpo](v5-sat-grpo.md) |
| **v6** HardNeg ★ | +GRPO·하드네거티브 | 1.0 | 0.8083 | 0.9456 | ★ **최종** | [v6-hardneg-grpo](v6-hardneg-grpo.md) |
| **v7** ColdStart | 콜드스타트SFT+추론GRPO | 0.7067 | 0.6733 | 0.7944 | ✗ FAIL | [v7-coldstart](v7-coldstart.md) |
| **v8a** Robust-A | 강건성GRPO rank32 | 1.0 | 0.8117 | 0.9411 | ✗ trade-off | [v8a-robust](v8a-robust.md) |
| **v8b** Robust-B | 강건성GRPO rank16 | 1.0 | 0.8067 | 0.9456 | ✗ 무이득(PASS) | [v8b-robust](v8b-robust.md) |

> 표 A(외부검증)와 표 B(v8_eval)는 **서로 다른 held-out 셋**이라 절대수치가 다릅니다(예: v6 OOD 0.855 vs 0.8083). 결론(서열·판정)은 동일.
> v7은 혼합 v8_eval 기준; OOD-only 셋에서는 0.42까지 붕괴. base(no-FT)는 balanced 0.85.

---

## 코드 인덱스 (실험 → 파일)

| 실험 | 데이터 빌드 | 학습 스크립트 | 병합 | 제출 CSV | HF 모델 |
|---|---|---|---|---|---|
| v4 | [`data_build/make_bbq_clean.py`](../data_build/make_bbq_clean.py) | [`train/unsloth/sft_unsloth.py`](../train/unsloth/sft_unsloth.py) | (16bit 병합) | — | `psh3333/…-v4-unsloth` |
| v5 | `make_bbq_clean.py` (BBQ 전체) | [`train/unsloth/grpo_unsloth.py`](../train/unsloth/grpo_unsloth.py) | — | `submissions/submission_v5.csv` | `…-v5-unsloth` |
| **v6** | [`data_build/mine_hard.py`](../data_build/mine_hard.py) | [`train/unsloth/grpo_hard_v6.py`](../train/unsloth/grpo_hard_v6.py) | — | `submissions/submission_v6_final.csv`(실이미지)·`submission_v6.csv`(텍스트) | **`…-v6`** |
| v7 | [`data_build/build_v7_data.py`](../data_build/build_v7_data.py) | [`train/unsloth/sft_reason_v7.py`](../train/unsloth/sft_reason_v7.py) + [`grpo_reason_v6.py`](../train/unsloth/grpo_reason_v6.py) | — | — | `…-v7` |
| v8a | [`data_build/build_v8_robust_pool.py`](../data_build/build_v8_robust_pool.py) + [`mine_v8_dynamic_pool.py`](../data_build/mine_v8_dynamic_pool.py) | [`train/unsloth/grpo_robust_v8.py`](../train/unsloth/grpo_robust_v8.py) | — | `submissions/submission_v8a.csv` | `…-v8a` |
| v8b | [`data_build/build_v8b_pool.py`](../data_build/build_v8b_pool.py) | [`train/unsloth/grpo_robust_v8b.py`](../train/unsloth/grpo_robust_v8b.py) | [`merge_v8b.py`](../train/unsloth/merge_v8b.py) | `submissions/submission_v8b.csv` | `…-v8b` |

공통 평가: [`inference/eval_suite.py`](../inference/eval_suite.py) · 독립셋 빌드 [`data_build/build_ood2.py`](../data_build/build_ood2.py) · 제출 [`inference/make_submission.py`](../inference/make_submission.py)
HF 네임스페이스: `psh3333/dacon-skku-bias-vlm-{v4-unsloth,v5-unsloth,v6,v7,v8a,v8b}` · W&B `psh243360-korea-university/dacon-bias`

> 제출 번들(2차 평가용 동결본)은 같은 코드를 [`submission/code/`](../submission/code/)에 복제합니다. 처음부터 재현하려면 [`submission/실행방법.md`](../submission/실행방법.md) §3.

---

## 더 읽기
- 방법론 상세: [`docs/EXPERIMENT_METHODOLOGY.md`](../docs/EXPERIMENT_METHODOLOGY.md)
- 전체 비교: [`docs/V4_V8B_COMPLETE_COMPARISON.md`](../docs/V4_V8B_COMPLETE_COMPARISON.md)
- 최종 모델 선택 근거: [`docs/FINAL_MODEL_SELECTION.md`](../docs/FINAL_MODEL_SELECTION.md)
- 재현성 검증(RTX A6000): [`submission/REPRODUCIBILITY_검증.md`](../submission/REPRODUCIBILITY_검증.md)
- DACON 점수: [`docs/DACON_SCORES.md`](../docs/DACON_SCORES.md)
