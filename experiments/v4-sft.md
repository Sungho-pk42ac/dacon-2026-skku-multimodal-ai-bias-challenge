# v4 — 단일토큰 SFT (기준선)

`기준선` · 이후 모든 RL 실험의 base merged 모델 · HF `psh3333/dacon-skku-bias-vlm-v4-unsloth`

## 가설 / 목적
공개 BBQ를 누출 제거·옵션 셔플·dedup 후 **가벼운 SFT**로 학습해, 출력을 **단일토큰 직답(0/1/2)** 으로 고정한다. 가장 단순·강건한 출발점을 만든다.

## 코드
| 단계 | 파일 |
|---|---|
| 데이터 빌드 | [`data_build/make_bbq_clean.py`](../data_build/make_bbq_clean.py) — 누출 SSOT(`val_ids.json`, sha1 sample_id), 시나리오 dedup, 옵션 셔플+라벨 재매핑, placeholder(336²) |
| 학습 | [`train/unsloth/sft_unsloth.py`](../train/unsloth/sft_unsloth.py) — Unsloth FastVisionModel, LoRA rank 32, 언어층 학습/비전 동결 |
| 평가 | [`inference/eval_suite.py`](../inference/eval_suite.py) |

```bash
python data_build/make_bbq_clean.py --out_dir data          # seed 42 결정론
python train/unsloth/sft_unsloth.py --model Qwen/Qwen3-VL-8B-Instruct \
    --train data/bbq_v4_train.json --out outputs/merged_v4 --epochs 3 --rank 32
```

## 데이터
- 소스: 공개 BBQ (Elfsong/walledai BBQ) — TRAIN 57,094 (시나리오 누출 0, 검증 실측).
- 이미지: 공개 텍스트데이터라 회색 placeholder(336×336). DACON 테스트 이미지는 미사용.

## 결과 (v8_eval 900)
| balanced(amb) | dis | OOD | 셔플 일관성 | 형식 유효율 | 속도 |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 0.8033 | 0.9478 | 0.9556 | 0.0586 |

## 결론
편향 balanced 1.0 포화·OOD/셔플 최강 기준선. **모든 RL의 base merged(`outputs/merged_v4`).** 단, GRPO가 아니므로 제출 모델은 아님 → v5~v8b가 여기서 분기.
