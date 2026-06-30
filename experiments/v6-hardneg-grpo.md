# v6 — HardNeg-GRPO ★ 최종 제출

`★ FINAL` · HF **`psh3333/dacon-skku-bias-vlm-v6`** · 공식 채점 경로 **실이미지 0.9628**

## 가설 / 목적
v5(포화)의 실패를 처방한다 — **v4가 틀리는 하드네거티브만** 모아 GRPO하면 보상 분산이 살아 정책이 실제로 개선되고, 일반추론 하드네거티브로 **OOD 일반화**까지 끌어올린다. 콜드스타트 SFT 없이 **GRPO-only(R1-Zero 동형)**.

## 코드
| 단계 | 파일 |
|---|---|
| 데이터 빌드 | [`data_build/mine_hard.py`](../data_build/mine_hard.py) — v4 오답 문항 채굴(SIQA·CSQA·OBQA·ARC) + BBQ-amb 기권 앵커, val_ids 전역 배제 |
| 학습 | [`train/unsloth/grpo_hard_v6.py`](../train/unsloth/grpo_hard_v6.py) — vLLM fast_inference GRPO, rank 32, G=4, 400 steps, lr 5e-6 cosine, 보상 2종(정답 +2.0/−0.5 · 형식 +0.5/−1.0) |
| 평가 | [`inference/eval_suite.py`](../inference/eval_suite.py) · 독립셋 [`data_build/build_ood2.py`](../data_build/build_ood2.py) |
| 제출 | [`inference/make_submission.py`](../inference/make_submission.py) |

```bash
# 1) 하드네거티브 풀
python data_build/mine_hard.py --base outputs/merged_v4 --out hardpool.json --val_ids data/val_ids.json --per 500
# 2) GRPO (최종 모델)
export WANDB_MODE=disabled
python train/unsloth/grpo_hard_v6.py --base outputs/merged_v4 --out outputs/merged_v6 \
    --hardpool hardpool.json --steps 400 --rank 32 --num_gen 4
# 3) 공식 제출 (실이미지 0.9628)
python inference/make_submission.py --model psh3333/dacon-skku-bias-vlm-v6 \
    --test_csv data/test.csv --image_root data/test/images \
    --sample_submission data/sample_submission.csv --max_pixels 262144 --out submission_v6_final.csv
```

## 결과
**표 A (외부검증, 결정축):** balanced 1.0 · s_AMB 0.0 · OOD **0.855** · ood2(미관측) **0.7783** · 셔플 0.9544 · image-invariance 0.975 — **6축 중 5축 best/tied**.
**표 B (v8_eval):** amb 1.0 · OOD 0.8083 · 셔플 0.9456.
**DACON 실채점(test 8,500):** 실이미지 **0.9628**(공식) · 텍스트집중 1.0(ablation).
- 편향 직접 증거: amb 사람선택오류 0.18→**0.0**, s_AMB 0.087→**0.0**.
- 재현성(RTX A6000): 추론 결정성 100%(2회 비트동일), 텍스트집중 제출본 99.29% 일치.

## 제출 산출물
- `submissions/submission_v6_final.csv` — 실이미지(공식, 0.9628)
- `submissions/submission_v6.csv` — 텍스트집중(ablation, 1.0)

## 결론
복잡도 변형 4종(v5/v7/v8a/v8b) 어느 것도 v6를 유의하게 넘지 못함 → **동급이면 가장 단순한 v6**(Occam). 최종 제출 확정. 근거: [`docs/FINAL_MODEL_SELECTION.md`](../docs/FINAL_MODEL_SELECTION.md).
