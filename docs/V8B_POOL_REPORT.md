# v8B 집중 풀 리포트

> 목표: v8A가 실패한 옵션순서 강건성만 정밀 타격(OOD 이득 보존). 공개데이터만, DACON 평가셋 미사용.
> 평가셋(v8_eval.json)·val_ids 항목은 풀에서 제외 → 학습/평가 분리.

- 채점 후보: **1560** (소스당 상한 260; bbq_amb/bbq_dis/ood_siqa/csqa/obqa/arc 각 260)
- 채굴: v4·v8A로 원본+셔플 채점 + v8A G=8 동적샘플링
- 태깅된 샘플: **208** / 선택(고유): **170** / 셔플짝 포함 풀: **340**

## 카테고리 분포 (선택 후, 상한 적용)
| 카테고리 | 개수 | 상한 |
|---|---:|---:|
| v8a_shuffle_inconsistent | 91 | 320 |
| both_wrong | 120 | 120 |
| v8a_regression | 7 | 160 |
| ood_v8a_gain | 2 | 120 |
| amb_person_error | 0 | 90 |
| dis_overabstain | 0 | 120 |
| paired_shuffle | 170 | — |

## 소스 분포 (풀)
| 소스 | 개수 | v8A 기준정확도 |
|---|---:|---:|
| ood_obqa | 112 | 0.7962 |
| ood_siqa | 106 | 0.8038 |
| ood_csqa | 100 | 0.8038 |
| ood_arc | 22 | 0.9615 |
| bbq_amb | 0 | 1.0 |
| bbq_dis | 0 | 1.0 |

→ **풀이 전부 OOD**: v4·v8A가 BBQ에서 완벽(1.0)이라 BBQ엔 회귀·과기권·사람오류가 없고,
동적샘플링상 too-easy(k=G)로 자동 제외됨. v8A가 잃은 셔플 일관성은 OOD에 집중되므로 OOD 타격이 정확한 처방.

## 동적 샘플링 (v8A, G=8)
- useful(0<k<G): **176** · easy(k=G): 1295 · hard(k=0): 89
- 풀은 useful 위주 카테고리(셔플불일치·양쪽오답)로 구성.

## variant 분포 (셔플 짝)
- orig: 170, shuf: 170
- 각 선택 샘플을 원본 + 셔플변형 2행으로 포함 → 순서불변(option-shuffle consistency) 직접 학습.

## BBQ 보존 전략 (풀에 BBQ 없음에 대한 처방)
- 저rank(16)·저lr(1e-6)·소수스텝(200)·attention-only로 v4의 BBQ 능력을 거의 흔들지 않음.
- eval 게이트(bbq_amb/dis acc, 과기권율)로 회귀 검증 → 결과 BBQ 1.0/1.0 유지 확인(회귀 없음).
