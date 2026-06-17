# FINAL v8B REPORT

> DACON 2026 성균관대 멀티모달 AI Bias 챌린지(236722). v8B = **v4 + GRPO-only + dynamic sampling
> + source-normalized reward + shuffle-recovery reward + single-token output**. 콜드스타트 SFT 없음,
> 추론확장 없음, v7 미사용. 공개데이터만(DACON 평가셋 미사용). 최종 라벨 = 모델 생성 텍스트.

## 1. v4 · v6 · v8A · v8B 비교표 (held-out v8_eval 900, 공개데이터)

| 지표 | v4 | v6 | v8A | **v8B** |
|---|---|---|---|---|
| BBQ ambiguous acc | 1.0 | 1.0 | 1.0 | **1.0** |
| BBQ disambiguated acc | 1.0 | 1.0 | 1.0 | **1.0** |
| balanced acc | 1.0 | 1.0 | 1.0 | **1.0** |
| **OOD acc** | 0.8033 | 0.8083 | 0.8117 | **0.8067** |
| **option-shuffle consistency** | 0.9478 | 0.9456 | 0.9411 | **0.9456** ✅ |
| unknown-position consistency | 1.0 | 1.0 | 1.0 | **1.0** |
| ambiguous person-selection error | 0.0 | 0.0 | 0.0 | **0.0** |
| disambiguated over-abstain rate | 0.0 | 0.0 | 0.0 | **0.0** |
| output format validity | 0.9556 | 0.9533 | 0.9733 | **0.9544** |
| inference speed (s/sample) | 0.0586 | 0.0597 | 0.0584 | **0.0582** |
| bias s_AMB / s_DIS | 0/0.04 | 0/0.04 | 0/0.04 | **0/0.04** |

OOD 소스별(v8B): arc 0.9533 · csqa 0.74 · obqa 0.76 · **siqa 0.7733**(v4 0.76→개선). 나머지는 v4와 동일.

**참고(전체 서사용)** — v7(콜드스타트 SFT)은 같은 eval에서 amb 0.7067 / OOD 0.6733 / shuffle 0.7944 /
person-error 0.2933 로 전방위 붕괴(특화 과적합). v5는 v4와 동급(GRPO 효과 0). 상세는 `V4_V7_SUMMARY.md`.

## 2. W&B
- run name: **v8B_robust_grpo**
- link: https://wandb.ai/psh243360-korea-university/dacon-bias/runs/xm3p2p81
- entity/project: `psh243360-korea-university` / `dacon-bias`
- 기록: loss + 보상 6종(answer/shuffle_consistency/abstain/source_normalized/format/length) 추이,
  config(rank·lr·steps·num_gen·temp·reward_weights·trainable_params·source분포), dynamic counts,
  per-source reward mean/std. 로컬요약: `outputs/wandb_summary_v8b.csv`, `docs/V8B_WANDB_SUMMARY.md`.

## 3. v8B 풀 요약 (`docs/V8B_POOL_REPORT.md`, `outputs/v8b_pool_stats.json`)
- 후보 1560(소스당 260, val_ids+v8_eval 제외) → v4·v8A 채점(원본+셔플) + v8A G=8 동적샘플링.
- 선택 170 고유 + 셔플짝 = **340행**. 구성: shuffle_inconsistent 91, both_wrong 120, regression 7, ood_gain 2.
- **전부 OOD**: v4·v8A가 BBQ에서 완벽(1.0)이라 BBQ는 동적샘플링상 too-easy(k=G)로 자동 제외 →
  v8A가 잃은 셔플 일관성은 OOD에 집중 → OOD 타격이 정확한 처방.
- dynamic: useful 176 / easy 1295 / hard 89 (G=8). source_base_acc(v8A): siqa/csqa 0.80, obqa 0.80, arc 0.96.

## 4. v8B 보상 설계 (`train/unsloth/grpo_robust_v8b.py`)
1. **answer** 정답 +2.0 / 오답 -1.0
2. **shuffle_consistency** 예측 옵션텍스트가 gold_text(순서무관)와 일치 +1.0 / 불일치 -1.0 (셔플짝과 결합 → 순서불변 학습)
3. **abstain** amb&pred==unk +0.5 / amb&pred!=unk -0.5 / dis 과기권 -0.6 / else 0
4. **source_normalized** (정답?1:0) − v8A기준정확도[src] → 소스별 zero-centered, 하드소스 상향
5. **format** 단일토큰 0/1/2 유효 +0.2 / 무효 -0.5 (v8A의 죽은신호 버그 수정)
6. **length** 단일토큰 길이 과다 -0.3
- config: base merged_v4, rank 16, lr 1e-6, steps 200, num_gen 8, temp 0.85, max_completion 4,
  vision frozen, MLP off(attention-only), bf16. wall 1710s. per-source answer reward: obqa+0.289/csqa+0.266/arc+0.140/siqa−0.059(std~1.45, 건강한 분산).

## 5. v8B 평가 결과 (`docs/V8B_EVAL_REPORT.md`, `outputs/eval_results_v8b.json`)
- OOD 0.8067(v4 0.8033↑, v8A 0.8117↓) · **셔플 일관성 0.9456 (v8A 0.9411 → 회복)** · BBQ 1.0/1.0 ·
  과기권 0.0 · 사람오류 0.0 · format 0.9544 · 속도 0.0582s.
- v8A가 만든 trade-off(OOD↑·셔플↓)에서 **셔플을 v6 수준으로 회복**하면서 OOD는 v4 이상 유지.

## 6. 분류: **PASS**
- 근거: OOD 0.8067 ≥ floor 0.8067 ✓ / 셔플 0.9456 ≥ 0.9450 ✓ / format 0.9544 ≥ (v4−0.02=0.9356) ✓ /
  셔플 v8A 대비 회복 ✓ / 과기권·사람오류 회귀 없음 ✓.
- STRONG_PASS(OOD≥0.8150 & 셔플≥0.9470) 미달 → PASS. (선호치엔 못 미치나 최소·안정 기준 모두 충족)

## 7. 최종 제출 CSV 경로
- `submissions/submission_v8b.csv` — **실제 테스트 이미지로 생성 예정**(아래 9.b 명령).
- 현재 상태: 실제 테스트 이미지 확보(`open/test/images`, 8500개) → pod 전송 후 생성. 검증은
  `submissions/submission_v8b_validation.json`(real_image_loaded_count=8500, placeholder_fallback_count=0, status OK 확인).

## 8. HF 패키지 경로
- `hf/v8b_package/` — adapter(74MB rank16) + README/model_card/inference_example/metadata/requirements/eval_summary.
- 기본 로컬 패키징(업로드는 `--push_to_hub` 명시 시, 기본 private).

## 9. 정확 재현 명령
**a. 평가(공개 held-out)**
```bash
python inference/eval_suite.py --model outputs/merged_v8b --tag v8b --fmt single --data /workspace/v8_eval.json
python inference/eval_report_v8b.py
```
**b. 제출 생성(실제 이미지)**
```bash
python inference/make_submission.py \
  --model outputs/merged_v8b \
  --test_csv data/test.csv \
  --sample_submission data/sample_submission.csv \
  --image_root data/test_images \
  --out submissions/submission_v8b.csv \
  --batch_size 8 --max_new_tokens 4 --seed 42
```
**c. HF 패키징**
```bash
python scripts/export_to_hf.py --model outputs/merged_v8b --adapter outputs/v8b_adapter \
  --out hf/v8b_package --version v8B --submission submissions/submission_v8b.csv
```
**d. 학습 재현**
```bash
python data_build/build_v8b_pool.py --v4 outputs/merged_v4 --v8a outputs/merged_v8a --eval /workspace/v8_eval.json
python train/unsloth/grpo_robust_v8b.py --base outputs/merged_v4 --pool /workspace/v8b_pool.json \
  --stats /workspace/v8b_pool_stats.json --out outputs/v8b_adapter --rank 16 --lr 1e-6 --steps 200 --num_gen 8 --temperature 0.85
python train/unsloth/merge_v8b.py --base outputs/merged_v4 --adapter outputs/v8b_adapter --out outputs/merged_v8b
```

## 10. 규칙 준수
`docs/DACON_RULE_COMPLIANCE_CHECKLIST.md` 참조 — 외부API 0, DACON 평가셋 미사용, 평가셋 패턴마이닝 0,
규칙기반 정답선택 0, 최종답=모델텍스트, train/inference 분리, 실제이미지 제출 강제(fallback 시 중단).
