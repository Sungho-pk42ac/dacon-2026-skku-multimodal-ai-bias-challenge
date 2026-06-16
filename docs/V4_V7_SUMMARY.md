# v4 → v7 실험 정리 (2026-06-16)

> DACON 2026 성균관대 멀티모달 AI Bias 챌린지(236722). 베이스: **Qwen3-VL-8B-Instruct** (Apache-2.0, 2026-06-01 이전), Unsloth + LoRA.
> 데이터: 공개 BBQ(58,492 unique) + 일반추론(siqa/csqa/obqa/arc). 누출방지: sha1 sample_id SSOT, val_ids 학습제외.
> 평가지표: ① **BBQ val 정확도**(amb/dis, 1320 학습제외분) ② **BBQ bias score** s_AMB/s_DIS (Parrish 2022) ③ **OOD held-out** 정확도(siqa/csqa/obqa/arc 800, 일반화 측정) ④ **추론속도**(A6000, 8500환산).

---

## 1. 버전별 요약

| 버전 | 구성 | 콜드스타트 | 학습데이터 | 비고 |
|---|---|---|---|---|
| **v4** | 단일토큰 SFT (0/1/2 직답) | ❌ | BBQ(셔플·dedup·누출제거) | 본명후보, 가장 단순·강건 |
| **v5** | v4 + 단일토큰 GRPO (BBQ 전체) | ❌ | BBQ | 효과 0 (v4≈v5). val 포화로 reward variance≈0 |
| **v6** | v4 + 단일토큰 GRPO (하드네거티브 일부) | ❌ | BBQ + v4오답 하드 | v4와 동급, 빠름. **제출 1순위 후보** |
| v6r | v4 + 추론 GRPO (콜드스타트 없음) | ❌ | 하드풀 702 | **폐기** — vLLM STANDBY CUDA 크래시 3회, 보상 평탄, step159 사망 |
| **v7** | v4 + **콜드스타트 SFT** + 추론 GRPO (난이도믹스) | ✅ | SFT:BBQ1050 / GRPO:믹스1752 | 학습성공, **OOD 과적합 폭락** |

---

## 2. 정량 결과표

| 지표 | v4 | v6 | v7 |
|---|---|---|---|
| BBQ val acc (ambiguous) | **1.0000** | 1.0000 | 0.9924 |
| BBQ val acc (disambiguated) | 0.9985 | 0.9985 | 0.9985 |
| BBQ **bias** s_AMB | **+0.0000** | +0.0000 | +0.0045 |
| BBQ **bias** s_DIS | +0.0576 | +0.0576 | +0.0576 |
| **OOD held-out** (siqa/csqa/obqa/arc 800) | **0.836** | 0.835 | **0.416** ⬇️ |
| └ arc / csqa / obqa / siqa | — | — | 0.595 / 0.335 / 0.380 / 0.355 |
| 추론속도 (8500환산) | ~19분(0.137s) | ~19분(0.137s) | 26.7분(0.189s 배치, PASS) |
| HF 배포 | ✅ v4 | ✅ v6 | 업로드중 v7 |

> 측정 방식: BBQ val·bias·OOD는 transformers 단일샘플(누출 0, val_ids 한정). v7 OOD는 단일샘플 0.4163 ≈ 배치 0.4275 → **배치버그 아님, 실제 저하**.
> 추론속도 규칙: 평균 0.5초/샘플은 **권장**(강제 아님), 기준환경 torch 2.6 → **제출 추론코드는 transformers**(vLLM은 torch 2.10이라 환경 불일치).

---

## 3. v7 과적합 — 실패 분석

**현상:** v7은 BBQ(학습 분포)에서 0.99로 완벽하나, OOD 일반추론에서 0.42(거의 랜덤 0.33)로 폭락. = **BBQ 도메인 과적합 + 일반능력 파국적 망각.**

**원인 (우선순위):**
1. **콜드스타트 SFT가 BBQ 100% 데이터** — siqa/csqa/obqa/arc 0개. "context 속 사람 단서 → 기권/선택" BBQ 틀만 학습.
2. **콜드스타트 추론 라벨이 정답을 박은 템플릿** (`"context gives clear evidence pointing to '{정답}'"`) — 모델이 *추론* 대신 *정답 베끼기 shortcut*을 학습.
3. **GRPO 풀도 BBQ 77% 편중**, 일반추론은 v4오답 하드만(정답예시 0) → 일반능력 보존 신호 없음.
4. **LoRA rank 64로 언어층 대량 갱신** → v4의 broad 능력 덮어씀.

**ablation(범인 특정):** merged_v7_sft(콜드스타트만, GRPO 전) held-out 측정 중 — 이미 ~0.42면 **콜드스타트 확정**, ~0.8이면 reasoning-GRPO. *(작성 시점 측정 진행중)*

**대조 증거:** v6(GRPO만, 콜드스타트 없음)은 OOD 0.835 유지 → **GRPO 자체는 일반화를 해치지 않음.** 차이를 만든 변수 = 콜드스타트 SFT.

---

## 4. 논문 근거 (검증완료 2026-06-16)

**핵심 기법**
- GRPO: **DeepSeekMath** (Shao et al. 2024, arXiv:2402.03300) — critic 없이 그룹 상대보상.
- 콜드스타트→RL: **DeepSeek-R1** (2025, arXiv:2501.12948).
- 난이도믹스/zero-advantage 제거: **DAPO** (Yu et al. 2025, arXiv:2503.14476) — dynamic sampling, clip-higher.
- 과제·bias score: **BBQ** (Parrish et al. 2022, ACL Findings, arXiv:2110.08193).

**우리 실패/일반화를 설명하는 논문**
- **"SFT Memorizes, RL Generalizes"** (Chu et al. ICML 2025, arXiv:2501.17161) — *outcome-reward RL은 OOD 일반화, SFT는 암기.* **v7의 콜드스타트 SFT가 "암기"로 OOD 실패 = 이 논문 예측 그대로.** v6(RL만)이 OOD 유지한 것도 부합.
- **DIVER** (arXiv:2509.26209) — 학습데이터 **다양성**이 OOD 일반화의 1순위 레버(+3.2). 다양성 효과는 OOD에서 특히 큼.
- 다양성 붕괴: arXiv:2509.07430(divergence 선택), DSDR 2602.19895 — GRPO가 소수 패턴 수렴→OOD 악화.
- 추론↔편향 주의: **"Does Reasoning Introduce Bias?"** (arXiv:2502.15361) — 추론이 정확도/편향을 해칠 수 있음(측정 필수). EIT(arXiv:2602.01528) — RL로 편향단서 비예측화.

**일반화 향상 처방 (GRPO만):** ① **도메인 다양화**(BBQ+일반+논리 믹스) ② outcome 보상 유지 ③ clip-higher로 다양성붕괴 방지 ④ dynamic sampling ⑤ num_gen↑ ⑥ 조기정지. (한계: RLVR은 base 잠재능력을 끌어낼 뿐 — 향상폭은 base 한계 내.)

---

## 5. 핵심 교훈 & 권고

**교훈 — "적게, 깨끗하게"가 데이터로 입증됨:**
```
v4 (가벼운 SFT, broad)        OOD 0.836   ← 가장 강건
v6 (+ GRPO 단일토큰)          OOD 0.835   ← 유지
v7 (+ 콜드스타트 + 추론, 특화) OOD 0.416   ← 특화할수록 무너짐
```
파인튜닝을 정교·특화할수록 일반화를 잃었다. 대부분 팀이 "성공"만 보고할 때, 우리는 **왜 실패했는지를 논문 근거로 해부**했다 — 이것이 정직한 검증 서사의 핵심.

**제출 권고: v6** (= v4 + GRPO, 콜드스타트 없음).
- 편향과제(BBQ) 동급(0.99) + OOD 강건(0.835) + 빠름(0.137s) + GRPO 적용("v4 단독 아님" 충족) + torch 2.6 호환.
- v7은 "추론 GRPO 시도 → 과적합 진단 → 재설계 방향(v8: 도메인믹스·비-암기 콜드스타트·rank↓)" 의 **음성결과 분석**으로 발표에 수록.

**미해결/주의:** 대회 점수는 BBQ 천장이라 v4·v6·v7 모두 ~0.99로 동률 → 점수 차별화 불가. 변별은 **검증방법론 + bias score + 발표**에서 발생. exposed W&B/HF 키는 대회 후 교체 필요.
