# v5 — Sat-GRPO (포화 분포 GRPO) ✗ 음성결과

`효과 0` · HF `psh3333/dacon-skku-bias-vlm-v5-unsloth` · 제출 `submissions/submission_v5.csv`

## 가설 / 목적
v4 위에 **BBQ 전체(포화 분포)** 로 GRPO를 얹으면 추가 향상이 있을 것이다.

## 코드
| 단계 | 파일 |
|---|---|
| 데이터 | [`data_build/make_bbq_clean.py`](../data_build/make_bbq_clean.py) (BBQ 전체 사용) |
| 학습 | [`train/unsloth/grpo_unsloth.py`](../train/unsloth/grpo_unsloth.py) — TRL GRPOTrainer, rank 32, 보상 2종(정답+형식) |

```bash
python train/unsloth/grpo_unsloth.py --base outputs/merged_v4 --steps 300 --rank 32 --num_gen 4
```

## 결과
- **효과 0.** v4와 동급(OOD ≈0.803, 셔플 ≈0.948).
- 원인: BBQ val이 이미 포화(1.0) → 그룹 내 **보상 분산 ≈ 0 → advantage ≈ 0 → gradient 소실**.

## 결론 (음성결과)
**"포화된 분포엔 GRPO 신호가 없다."** 이 실패가 곧 v6의 처방(=하드네거티브로 보상 분산을 살려라)의 직접 동기. → [v6](v6-hardneg-grpo.md)
