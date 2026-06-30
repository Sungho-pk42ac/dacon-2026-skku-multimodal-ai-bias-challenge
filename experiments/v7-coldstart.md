# v7 — ColdStart SFT + 추론 GRPO ✗ FAIL (음성결과)

`FAIL · 전방위 붕괴` · HF `psh3333/dacon-skku-bias-vlm-v7` · 제출 안 함

## 가설 / 목적
DeepSeek-R1식 **콜드스타트(정답 박은 추론 템플릿 SFT) + 추론 GRPO**가 일반화를 도울 것이다. (R1의 distill-then-RL 재현 시도)

## 코드
| 단계 | 파일 |
|---|---|
| 데이터 | [`data_build/build_v7_data.py`](../data_build/build_v7_data.py) — BBQ 100% 콜드스타트 + 추론 템플릿 |
| 학습 | [`train/unsloth/sft_reason_v7.py`](../train/unsloth/sft_reason_v7.py) (콜드스타트 SFT) → [`grpo_reason_v6.py`](../train/unsloth/grpo_reason_v6.py) (추론 GRPO), LoRA rank 64 |

## 결과 (v8_eval 900)
| balanced(amb) | OOD | 셔플 | amb 사람선택오류 | s_AMB |
|---|---|---|---|---|
| **0.7067** | **0.6733** | **0.7944** | **0.2933** | **−0.067** (역편향) |

- OOD-only 셋에서는 **0.42**까지 붕괴.
- **ablation으로 범인 확정:** 콜드스타트 SFT만으로 (GRPO 이전에) 이미 OOD가 붕괴 → **GRPO 무죄, 콜드스타트 SFT가 파국적 망각의 원인.**

## 결론 (음성결과 — 가장 중요한 발견)
Chu et al. 2025 *"SFT Memorizes, RL Generalizes"* 예측 그대로. **"특화·암기할수록 일반화가 붕괴한다."** v6/v8a/v8b의 GRPO-only가 일반화를 보존하는 것과 대조 → 발표의 핵심 서사(§6.2)를 떠받치는 증거.
