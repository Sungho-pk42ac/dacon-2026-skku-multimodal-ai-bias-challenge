# v8a — Robust-GRPO 1차 (rank 32) ✗ 음성결과 (trade-off)

`OOD↑ ↔ 셔플↓` · HF `psh3333/dacon-skku-bias-vlm-v8a` · 제출 `submissions/submission_v8a.csv`

## 가설 / 목적
콜드스타트를 빼고(GRPO-only) **강건성 인지 보상 5종**을 도입하면 OOD와 옵션순서 강건성을 동시에 올릴 수 있다.

## 코드
| 단계 | 파일 |
|---|---|
| 데이터 빌드 | [`data_build/build_v8_robust_pool.py`](../data_build/build_v8_robust_pool.py) + 동적샘플링 [`mine_v8_dynamic_pool.py`](../data_build/mine_v8_dynamic_pool.py) |
| 학습 | [`train/unsloth/grpo_robust_v8.py`](../train/unsloth/grpo_robust_v8.py) — rank 32, 보상 5종 |

## 결과 (v8_eval 900)
| balanced | OOD | 셔플 일관성 |
|---|---|---|
| 1.0 | **0.8117 (최고치)** | **0.9411 (하락)** |

- OOD는 전 버전 최고치 달성. 그러나 **옵션순서 셔플 일관성이 0.9411로 회귀** — OOD↑를 강건성↓와 맞바꾼 **trade-off**.
- format 보상이 −0.5 고정 버그(죽은 신호) → reward 평균이 음수였으나 eval은 정상(보상 모니터링의 함정).

## 결론 (음성결과)
"OOD만 보면 최고"지만 강건성 회귀로 **제출 부적합**. 이 trade-off를 해소하려는 시도가 [v8b](v8b-robust.md). v6 대비 명확한 우위 없음 → 최종 제출 탈락.
