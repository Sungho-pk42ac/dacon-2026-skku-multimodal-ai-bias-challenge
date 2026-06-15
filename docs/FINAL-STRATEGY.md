# 최종 단일 전략 — 멀티에이전트 회의 합의안 (2026-06-15)

> 회의체: asset-scout · oss-researcher-A · oss-researcher-B · repo-auditor · proposer · data-risk · devils-advocate
> 근거 문서: `.omc/research/{asset-inventory, oss-research-A, oss-research-B, repo-audit, debate-proposer, debate-data-risk, debate-critic}.md`
> 결론 한 줄: **"적게, 깨끗하게(less, but clean)" — 검증된 6개 레버에 집중, 화려한 기법은 발표용 +α로 강등.**

---

## 0. 회의가 뒤집은 전제 (가장 중요)

**현재 자체검증 BA 0.97은 val 누출로 부풀려진 값(INFLATED).**
- `data-risk`가 코드로 입증: `bbq_val.jsonl`(make_bbq_data.py:108–116, 카테고리별 셔플)과 `bbq_reason_train.json`(make_bbq_reasoning.py:103–108, 글로벌 셔플)은 분할축·RNG 소비순서가 다르고, reasoning 생성기에 **val 배제 코드가 아예 없으며**, 양쪽 다 `sample_id`가 없어 중복 검출조차 불가.
- ⇒ v2 SFT가 GPT-CoT와 함께 **암기한 행을 다시 채점**한 것. 일반화 지표로 신뢰 불가.
- ⇒ **누출 제거 + 재측정 전엔 v2/v3/v4/GRPO 어느 것도 점수로 비교 금지.** (BLOCKER)

현장 증거: v3 submission 실측 **1.65초/샘플** = 규칙 0.5초의 3.3배. 속도가 한 번도 실측 안 된 채 계획에 박혀 있었음을 입증.

---

## 1. Shippable Core — 검증된 6개 레버 (DO THESE)

| # | 레버 | 효과 | 비용 |
|---|---|---|---|
| 1 | **val 누출 제거 → 정직한 BA** | 단일 최대 레버(나머지 선택의 전제) | 반나절 |
| 2 | **2단계 SFT: CoT→직답 self-distillation** (base 아닌 **merged_v2**에서 출발) | 추론능력 보존 + 속도, 저위험 고가치 | 2~3일 |
| 3 | **단일토큰 직답 타깃("0"/"1"/"2")** | 속도(0.5초)+파싱안정 동시, 12토큰 잘림 논쟁 종결 | 0.5일 |
| 4 | **entropy 임계 기권 게이트** (forward 1회) | BA의 ambiguous 50% 비중 안전 확보 | 1일 |
| 5 | **실제 test 이미지 추론 A/B** (학습주입 X) | 이득 양수면 채택, 저비용 | 반나절 |
| 6 | **2-fold + 카테고리 분해 검증** | 단일 val 과적합 방지, 정직한 선정 | 1일 |
| + | **vLLM 전환 + 이미지토큰 512** | 0.5초 1차 수단 | 1~2일 |
| + | **DeCAP 모호성 프롬프트** (프롬프트만) | 저비용 노벨티 | 1일 |

## 2. 실행 순서 (의존성)

- **D0 (BLOCKER)**: ①val 누출 제거(sha1 content-hash로 `sample_id` → `val_ids.json` SSOT → 모든 생성기 강제배제+assert) + 깨끗한 BA 재측정 / **0.5초 A6000 실측(현 모델 vLLM 전환만)** / max_new_tokens 제출경로 하드코딩(현 버그: default 96).
- **D1**: 단일토큰 직답 타깃(②③) + 실제 이미지 추론 A/B(⑤) — A/B가 "이미지 중요"로 나오면 비전타워 동결 해제까지 연쇄 재설계 필요하므로 **가장 먼저 분기 확정**.
- **D2~5**: 2단계 증류 SFT(②) + DeCAP 프롬프트.
- **D5~7**: entropy 기권 게이트(④) — 깨끗한 val에서만 threshold 튜닝.
- **D7~10**: 2-fold+카테고리 분해(⑥)로 정직한 모델 선택. OOD proxy(StereoSet/WinoBias)는 **1회 tiebreaker만**(반복비교=overfit).
- **D10~14**: 버퍼(재시도·발표). 여유 시 GRPO/앙상블/AWQ를 발표 +α.

## 3. 현 v2/v3/v4 대비 변경점

- **v3**(4ep, `<think>`, 1.65초): **폐기 후보** — 속도 하드게이트 위반(2차 코드검증 시 실행시간 초과 위험).
- **v4**(직답): 방향은 맞으나 두 가지 수정 — (a) base가 아니라 **merged_v2에서 출발(증류)**, (b) 타깃을 `"0) 답"`이 아니라 **단일 토큰 "0"**.
- **검증**: `bbq_val` 누출 제거 후 v2/v4 전부 재측정 → 그 위에서만 비교.

## 4. KILL 목록 (점수 증거 약함 → 발표용으로만)

GRPO 보상 5종 재설계 · 데이터 합성 10%(BBQ paraphrase=eval 모방 회색지대) · 외부소스 학습주입(StereoSet/WinoBias/VLBiasBench) · self-consistency N≥3(속도 위반) · multi-persona(다중호출) · 베이스 교체(InternVL3) · multi-LoRA 앙상블 · AWQ(0.5초를 다른 수단으로 못 맞출 때만 조건부).

## 5. 규칙 준수 체크

- 오프라인 추론(API 0) ✅ / 모델 6/1 이전(Qwen2.5-VL-7B 유지) ✅ / 학습·추론 분리 ✅ / 0.5초(단일토큰+vLLM로 확보) ✅
- **No Leakage**: BBQ paraphrase 합성·외부 학습주입 전부 KILL → 회색지대 회피. BBQ는 train split만, 표면형 복제 없이.
- **최종답 LLM 생성**: 기권 게이트는 "3선택지 중 재배치"로 한정(외부 규칙이 새 답 생성 금지) → 발표 방어논리 확보.

## 6. 핵심 메시지

**Private/Hidden은 기법 수가 아니라 검증의 정직성이 결정한다.** 솔로가 9개를 어설프게 치면 누출된 직관으로 최종 제출을 골라 Private에서 무너진다. 6개만 깨끗하게.
