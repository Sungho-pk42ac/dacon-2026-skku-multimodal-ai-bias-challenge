# v8 Research Plan — Robustness-aware GRPO (2026-06-16)

> 목표: **v4 SFT의 강건성을 보존하면서**, robustness-aware GRPO가 일반화·기권·옵션순서 강건성을 *개선하는지* 검증.
> v7의 실패(과적합)를 교정. 신규 기법 남발 금지, 재현성 > 신규성.

## 0. 왜 v4가 안정성 베이스라인인가
실측(docs/V4_V7_SUMMARY.md): v4 OOD 0.836 / BBQ 0.99. 가장 단순(단일토큰 SFT)하면서 가장 강건. v6(=v4+GRPO)도 OOD 0.835로 동급 유지 → **GRPO 자체는 일반화를 해치지 않음**(SFT Memorizes/RL Generalizes, Chu 2025, 2501.17161).

## 1. 왜 v7이 실패했나 (교정 대상)
v7 OOD 0.42 폭락. 교란요인 다수: ① **강한 콜드스타트 SFT가 BBQ 100%** ② **정답을 박은 rationale(shortcut 학습)** ③ context/question을 q로 합친 포맷 불일치 ④ **LoRA rank 64**(언어층 과갱신) ⑤ 추론 GRPO. → v8은 이 다섯을 **모두 제거/완화**.

## 2. v8 핵심 설계 결정 (v7 대비)
| 요인 | v7 (실패) | v8 (교정) |
|---|---|---|
| 베이스 | v4 + 콜드스타트 | **outputs/merged_v4 직접** (콜드스타트 X) |
| 포맷 | q에 context+question 병합 | **context/question 분리** (통합 스키마) |
| LoRA rank | 64 | **16~32** |
| 파인튜닝 대상 | language+attn+MLP | **attention 우선, vision 동결, MLP 보류** |
| 학습데이터 | BBQ 77% | **도메인 다양화**(BBQ+OOD+UnQover, DIVER 2509.26209) |
| 보상 | accuracy+format | **강건성 보상 6종**(기권·셔플·반고정관념·형식·길이) |
| 선택 | 무조건 사용 | **게이트 통과 시에만 v4 대체** |

## 3. 가설
accuracy-only GRPO가 v4를 못 넘은 건 보상신호가 포화/약했기 때문. 보상이 **명시적으로** ①기권 일관성 ②옵션셔플 일관성 ③OOD 보존 ④반고정관념 ⑤형식 안정 ⑥실이미지 사용을 겨냥하면 **강건성**은 개선될 수 있다(점수보다 강건성·발표가치).

## 4. 데이터 거버넌스 (누출 방지 — 엄수)
- **DACON test/private/hidden 데이터는 최종 추론에만** 사용. 학습/검증/프롬프트튜닝/규칙/합성에 일절 사용 금지.
- 평가셋 패턴 추론 금지.
- 학습·검증은 **독립 공개 데이터**(BBQ via Elfsong/walledai, siqa/csqa/obqa/arc, UnQover)만.
- val_ids(SSOT)로 학습/평가 분리. 모든 공개 출처·전처리 기록(아래 §7).

## 5. 통합 스키마 (Phase 2)
```json
{"src":"bbq_amb|bbq_dis|ood_siqa|ood_csqa|ood_obqa|ood_arc|unqover|public_vqa",
 "context":"...", "question":"...", "answers":["a","b","c"],
 "label":0, "unknown_idx":2, "image_path":null, "meta":{}}
```
규칙: context/question 분리, answers 길이 3, label∈{0,1,2}, 기권옵션 있으면 unknown_idx 설정.

## 6. 보상 설계 (Phase 6, grpo_robust_v8.py)
6종, 각 컴포넌트 개별 로깅:
1. answer (+2.0/−1.0) 2. format (+0.3/−0.5) 3. abstain_consistency (amb→unknown +1.0/−1.0, dis 과기권 −0.5)
4. option_shuffle_consistency (셔플쌍 의미일치 +0.7/−0.7) 5. stereotype_penalty (인구통계 추측 −0.5, 보수적) 6. length_penalty (단일토큰 짧게 선호).
투기적 보상 남발 금지. v8A=단일토큰(max_completion 4), v8B=evidence-format(16~32)만 선택적.

## 7. 모델 선택 게이트 (Phase 8) — v8이 v4를 대체하는 조건(전부 충족)
- BBQ amb acc ≥ v4 − 0.001 · BBQ dis acc ≥ v4 − 0.001 · OOD ≥ v4 − 0.01
- 옵션셔플 일관성 v4 대비 ↑ · ambiguous 사람선택 오류 ↓ · 속도 규칙내 · 오프라인 외부API 0
- 하나라도 실패 → **최종 제출 v4 유지**, v8은 ablation/연구결과로 보고.

## 8. 산출물 (deliverables)
build_v8_robust_pool.py · mine_v8_dynamic_pool.py · grpo_robust_v8.py · infer_unified.py · eval_suite.py · V8_RESEARCH_PLAN.md(본) · V8_RESULTS.md · V4_V8_EVAL_TABLE.md · 최종권고.

## 9. 실행 단계
P1 재평가(v4/v5/v6/v7/+v7sft/+v7-GRPO-only) → P2 스키마 → P3 robust pool → P4 infer_unified → P5 eval_suite → P6 grpo_robust_v8(v8A 먼저) → P7 dynamic sampling(오프라인 우선) → P8 게이트 → P9 문서 → HF배포 + W&B.

## 논문 근거
GRPO(DeepSeekMath 2402.03300) · 콜드스타트 위험(R1 2501.12948) · 다양성(DIVER 2509.26209) · SFT암기/RL일반화(Chu 2501.17161) · DAPO dynamic sampling/clip-higher(2503.14476) · BBQ bias(Parrish 2110.08193) · 추론↔편향 주의(2502.15361).
