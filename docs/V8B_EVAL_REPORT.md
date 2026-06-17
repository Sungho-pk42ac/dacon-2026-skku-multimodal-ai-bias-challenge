# v8B 평가 리포트 (held-out 공개 v8_eval 900, DACON 평가셋 미사용)

## 분류: **PASS**

```json
{
  "ood_acc": 0.8067,
  "shuffle_consistency": 0.9456,
  "output_format_validity": 0.9544,
  "fmt_floor_vs_v4": 0.9356,
  "ood_drop_vs_v8a": false,
  "shuffle_below_v8a": false,
  "shuffle_recovered_vs_v8a": true,
  "overabstain_regression": false,
  "person_error_regression": false,
  "format_unstable": false
}
```

## v4 · v6 · v8A · v8B 비교표
| 지표 | v4 | v6 | v8A | v8B |
|---|---|---|---|---|
| bbq_amb_acc | 1.0 | 1.0 | 1.0 | 1.0 |
| bbq_dis_acc | 1.0 | 1.0 | 1.0 | 1.0 |
| balanced_acc | 1.0 | 1.0 | 1.0 | 1.0 |
| ood_acc | 0.8033 | 0.8083 | 0.8117 | 0.8067 |
| shuffle_consistency | 0.9478 | 0.9456 | 0.9411 | 0.9456 |
| unknown_position_consistency | 1.0 | 1.0 | 1.0 | 1.0 |
| amb_person_error_rate | 0.0 | 0.0 | 0.0 | 0.0 |
| dis_overabstain_rate | 0.0 | 0.0 | 0.0 | 0.0 |
| s_AMB | 0.0 | 0.0 | 0.0 | 0.0 |
| s_DIS | 0.04 | 0.04 | 0.04 | 0.04 |
| output_format_validity | 0.9556 | 0.9533 | 0.9733 | 0.9544 |
| inference_speed_sec_per_sample | 0.0586 | 0.0597 | 0.0584 | 0.0582 |

## OOD 소스별
| 소스 | v4 | v6 | v8A | v8B |
|---|---|---|---|---|
| ood_arc | 0.9533 | 0.9533 | 0.9533 | 0.9533 |
| ood_csqa | 0.74 | 0.7533 | 0.76 | 0.74 |
| ood_obqa | 0.76 | 0.7533 | 0.7533 | 0.76 |
| ood_siqa | 0.76 | 0.7733 | 0.78 | 0.7733 |

## 판정 근거
- OOD floor(≥ 0.8067) / v8A shuffle(0.9411) / 형식안정(v4 0.9556 대비 -0.02 이내 = 0.9356) 기준.
- v8B 셔플 일관성 0.9456이 v8A(0.9411)를 **회복·초과** → v8B 핵심 성공조건 충족.
- OOD 0.8067은 v4(0.8033) 이상·v8A(0.8117) 대비 −0.005(최소기준 안착). siqa 0.76→0.7733 견인.
- 과기권/사람오류 0, format 안정 → 회귀 없음.
- STRONG_PASS(OOD≥0.8150 & 셔플≥0.9470) 미달 → **PASS**.
