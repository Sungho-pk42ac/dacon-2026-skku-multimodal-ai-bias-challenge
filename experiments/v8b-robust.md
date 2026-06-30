# v8b — Robust-GRPO 2차 (rank 16) ✗ 음성결과 (PASS이나 무이득)

`trade-off 해소·무이득` · HF `psh3333/dacon-skku-bias-vlm-v8b` · 제출 `submissions/submission_v8b.csv`

## 가설 / 목적
v8a의 **OOD↑↔셔플↓ trade-off를 해소** — OOD를 유지하며 옵션순서 강건성을 회복한다. (초기엔 "메인 후보"였으나 외부검증 후 v6로 최종 결정)

## 코드
| 단계 | 파일 |
|---|---|
| 데이터 빌드 | [`data_build/build_v8b_pool.py`](../data_build/build_v8b_pool.py) — v4·v8a 채점으로 채굴(셔플불일치 91 + 양쪽오답 120 + 회귀 7 + OOD이득 2 = 170고유 ×셔플짝 = **340행**, 전부 OOD) |
| 학습 | [`train/unsloth/grpo_robust_v8b.py`](../train/unsloth/grpo_robust_v8b.py) — rank 16, lr 1e-6, attention-only, **보상 6종**, 200 steps |
| 병합 | [`train/unsloth/merge_v8b.py`](../train/unsloth/merge_v8b.py) |

처방 6가지: ① 저rank(16)·저lr(1e-6) ② 셔플짝 데이터증강 ③ shuffle-consistency 보상(정답텍스트 순서무관 일치) ④ source-normalized 보상 ⑤ format 버그 수정 ⑥ 동적샘플링(useful 위주).

```bash
python data_build/build_v8b_pool.py --v4 outputs/merged_v4 --v8a outputs/merged_v8a --eval /workspace/v8_eval.json
python train/unsloth/grpo_robust_v8b.py --base outputs/merged_v4 --pool v8b_pool.json \
    --out outputs/v8b_adapter --rank 16 --lr 1e-6 --steps 200 --num_gen 8 --temperature 0.85
python train/unsloth/merge_v8b.py --base outputs/merged_v4 --adapter outputs/v8b_adapter --out outputs/merged_v8b
```

## 결과
**표 B (v8_eval):** balanced 1.0 · OOD 0.8067 · 셔플 **0.9411→0.9456 회복** · 판정 **PASS**(STRONG 미달).
**표 A (외부검증):** OOD 0.8517 · ood2 0.775 · 셔플 0.9522 · image-invariance 0.9717.
**DACON 실채점:** 실이미지 0.9643.

## 결론 (음성결과)
강건성 보상·셔플짝·소스정규화로 **v8a의 회귀를 되돌릴 수 있음을 입증**(방법론적 정점). 그러나 v6 대비 net 이득은 노이즈 수준(셔플 +0.005, ood2 −0.003, DACON Δ0.0015) → **동급이면 더 단순한 v6**. 서사적 기여는 크나 제출 모델은 [v6](v6-hardneg-grpo.md).
