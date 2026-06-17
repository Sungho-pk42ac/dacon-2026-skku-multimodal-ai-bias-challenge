# 최종 모델 선택 — 외부검증 기반 결정 (2026-06-17)

## 결정: **v6 최종 제출** (1·2차)

> 본 결정은 **외부검증(eval-only, 학습 없음)** 결과로 갱신됨. 이전 초안(v8B 권고)을 대체.
> 근거 데이터: `docs/EXTERNAL_VALIDATION_REPORT.md`, `outputs/external_validation_results.json`,
> `outputs/external_validation_table.csv`, `outputs/eval_results_v4_v8.json`. W&B `external_validation_v6_v8a_v8b`.

## 결정규칙 적용 (사용자 지정)
- "v6를 best/tied(일반화·셔플·편향·속도)면 선택" → **충족**.
- "v8A는 외부 VQA/GQA/AOKVQA에서 v6를 **명확히** 능가하고 편향강건성 유지할 때만" → 미충족(v8a가 v6를 명확히 능가하는 항목 없음, ood2 비후보).
- "v8B는 편향/셔플 강건성에서 v6를 **명확히** 능가하고 OOD 손실 없을 때만" → 미충족(셔플 +0.005 노이즈수준, OOD는 v6가 더 높음).
- "혼조면 v6" → **해당. 결과: v6.**

## 증거 요약
| 결정축 | base(no-FT) | v6 | v8a | v8b | 판정 |
|---|---|---|---|---|---|
| 편향 balanced (v8eval) | 0.85 | **1.0** | 1.0 | 1.0 | 파인튜닝 필수 / v6 tied-best |
| 편향 s_AMB (낮을수록↑) | 0.0867 | **0.0** | 0.0 | 0.0 | v6 tied-best |
| 인분포 OOD (v8eval) | 0.8467 | **0.855** | 0.8517 | 0.8517 | **v6 최고** |
| 독립 일반화 (ood2, 미관측) | 0.705 | **0.7783** | (비후보) | 0.775 | **v6 최고** |
| 셔플 일관성 (v8eval) | 0.9122 | **0.9544** | 0.9544 | 0.9522 | v6 tied-best |
| image-invariance (ood2) | 0.8967 | **0.975** | — | 0.9717 | **v6 최고** |
| 속도 s/샘플 (v8eval) | 0.1074 | 0.0778 | 0.0764 | 0.0767 | 전부 제한 충족 |

**핵심:** v6는 6개 결정축 중 5개에서 best/tied-best. v8a/v8b 어느 것도 v6를 명확히 능가하지 못함.
파인튜닝 모델이 비파인튜닝 base를 **독립 미관측셋에서도** 능가 → 일반화 손실 없음(v7식 망각 없음).

## 권고 출력 (사용자 요청 항목)
- **best model recommendation: `v6`**
- **exact model path**:
  - pod: `/workspace/models/v6`
  - HF: `psh3333/dacon-skku-bias-vlm-v6`
  - 로컬(재머지 필요시): `outputs/merged_v6`
- **HF packaging: `v6` 사용** (v8A/v8B 아님). v8a/v8b/v7는 실험·음성결과 아카이브로 유지.

## DACON 제출 CSV 생성 명령 (모델 확정 후 — 이제 실행 가능)
실제 테스트 이미지로 추론(사용자 이미지규칙: DACON 최종추론은 실이미지 필수, placeholder fallback>0이면 중단):
```bash
cd /workspace/dacon-bias-challenge
python3 inference/make_submission.py \
  --model /workspace/models/v6 \
  --test_csv /workspace/data/test.csv \
  --image_root /workspace/data \
  --out submissions/submission_v6_final.csv \
  --max_pixels 262144        # 512^2 캡(심사 0.5s/샘플 한도 내, ~0.24s/샘플)
# real_image_loaded_count==전체 & placeholder_fallback_count==0 확인(아니면 자동 exit 2).
```
> placeholder 버전(텍스트전용/이미지없음 케이스 대조용)은 이미 `submissions/submission_v8b_placeholder.csv` 존재.
> v6 placeholder 대조가 필요하면 `--image_root /workspace/nonexistent --allow_placeholder` 추가.

## 폴백
v6 추론환경 비호환 시 → v8b(동급, 강건성보상 서사 보유). v8a는 셔플 회귀로 비권장. v4는 GRPO 아님(방침상 최후수단).

## 주의 — 변별력의 본질
v6·v8a·v8b는 BBQ 천장(1.0)과 OOD 노이즈(±수문항)로 **점수상 사실상 동급**. v6 선택의 실질 근거는
"동급이면 더 단순하고 검증이력이 풍부하며 독립셋에서 미세우위인 모델"이라는 보수적 원칙. 논문에서는
v8a/v8b/v7를 강건성·음성결과 분석으로 함께 보고(서사적 기여는 v8B 파이프라인이 정점, 제출모델은 v6).
