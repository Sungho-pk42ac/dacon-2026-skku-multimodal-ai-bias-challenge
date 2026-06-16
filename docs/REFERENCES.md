# 참고문헌 (검증완료 2026-06-16, WebSearch 확인)

> v6/v6r(GRPO)·v7(개선설계)·평가지표의 방법론적 근거. 인용정보는 arXiv/ACL 원문 확인.

## 1. GRPO 알고리즘 (v5·v6·v6r 학습의 핵심)
**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**
Shao et al., 2024. arXiv:2402.03300. https://arxiv.org/abs/2402.03300
- **Group Relative Policy Optimization(GRPO)** 최초 제안. PPO 변형이되 **value/critic 모델 없이** 한 프롬프트에 대한 *그룹 내 상대적 보상*으로 baseline(advantage)을 추정 → 메모리·연산 절감 + 학습 안정.
- 우리 적용: v5/v6(단일토큰)·v6r(추론) 모두 이 GRPO로 학습. reward = accuracy(+format).

## 2. SFT 콜드스타트 → RL 레시피 (v7 설계 근거)
**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
DeepSeek-AI, 2025. arXiv:2501.12948. (Nature, s41586-025-09422-z) https://arxiv.org/abs/2501.12948
- **R1-Zero**: SFT 없이 순수 RL만으로도 추론(CoT·자기검증) 창발 가능함을 보임.
- **R1(풀버전)**: 소량의 **cold-start SFT로 추론 형식을 먼저 가르친 뒤 RL** → 안정·성능 향상.
- 우리 적용: v7에서 "쉬운 문제로 추론형식 SFT 콜드스타트 → GRPO" = R1(풀) 레시피. (현 v6r은 콜드스타트 생략이라 형식·정책 동시학습으로 버거움 = R1-Zero에 가까운 어려운 셋업.)

## 3. 난이도 믹스 / zero-advantage 제거 (v7 핵심 개선 근거)
**DAPO: An Open-Source LLM Reinforcement Learning System at Scale**
Yu et al., ByteDance Seed & Tsinghua AIR, 2025. arXiv:2503.14476 (NeurIPS 2025). https://arxiv.org/abs/2503.14476
- 4기법 중 **Dynamic Sampling**: 한 프롬프트의 생성 답이 *전부 정답 또는 전부 오답*이면 advantage=0 → gradient 없음 → 그런 프롬프트를 걸러/리샘플해 학습 효율·안정 향상.
- **우리 문제와 정확히 일치**: v6r 하드풀은 "v4가 틀린 것만" 모아 *전부 오답 그룹*이 많아 신호가 죽음(answer 보상 평탄). → v7은 **풀리는 문제 + 어려운 문제 믹스**(=advantage가 살아있는 프롬프트 확보)로 교정. 그 외 Clip-Higher, Token-level PG loss, Overlong reward shaping도 참고.

## 4. 과제·편향지표 (BBQ, s_AMB/s_DIS)
**BBQ: A Hand-Built Bias Benchmark for Question Answering**
Parrish, Chen, Nangia, Padmakumar, Phang, Thompson, Htut, Bowman, 2022. Findings of ACL 2022. arXiv:2110.08193. https://arxiv.org/abs/2110.08193
- 9개 사회 카테고리, **58,492 unique 예제**(우리 스캔과 일치). ambiguous(정답=기권) vs disambiguated 2단계.
- **Bias score**: 모델이 정답이 고정관념과 *일치*할 때 평균 최대 3.4%p(성별은 5%p+) 더 잘 맞힘 = 편향. 우리 `bias_score_eval`의 s_AMB/s_DIS 정의 원천.

---
### 보조 근거
- Curriculum Learning (Bengio et al., 2009) — 쉬움→어려움 순 학습(v7 난이도 믹스의 고전 근거).
- UnQover (Li et al., 2020) — underspecified 편향 진단(우리 v4 100% 기권 검증에 사용).

---
## 최신 동향 (2025~2026, WebSearch 확인) — v7 직접 관련

### ★ 우리 접근과 거의 동일 (강력한 정당화)
**Making Bias Non-Predictive: Training Robust LLM Reasoning via Reinforcement Learning** (2026)
arXiv:2602.01528. https://arxiv.org/abs/2602.01528
- **Epistemic Independence Training(EIT)**: RL로 **편향 단서를 보상에 비예측적(non-predictive)으로** 만들어 추론모델의 공정성 확보.
- = 우리 v7(추론 GRPO로 "근거 없으면 기권" 강화)과 같은 철학. **"RL로 편향을 줄인다"가 2026 최신 연구 방향임을 입증.**

### ⚠️ 중요한 반례 (정직하게 균형) — 우리 "측정주의"를 정당화
**Does Reasoning Introduce Bias? Social Bias Evaluation and Mitigation in LLM Reasoning** (2025)
arXiv:2502.15361. https://arxiv.org/abs/2502.15361
- **CoT/추론이 정확도엔 좋지만 고정관념을 오히려 표면화·악화시킬 수 있음.**
**Inference-Time Reasoning Selectively Reduces Implicit Social Bias in LLMs** (2026)
arXiv:2602.04742. https://arxiv.org/abs/2602.04742
- 추론이 암묵편향을 줄이나 **모델 의존적**(항상은 아님).
→ 시사점: **"추론=편향감소"를 가정하지 말고 반드시 측정해야 한다.** 우리가 bias_score(s_AMB/s_DIS)를 held-out에서 실측하는 설계가 바로 이 교훈의 실천.

### GRPO 이론·개선 (v7 하이퍼파라미터 근거 보강)
- **RLVR: GRPO's Effective Loss, Dynamics, and Success Amplification** (2025) arXiv:2503.06639 — GRPO 손실·동역학 이론.
- **Fairness Reward Models** (2025) arXiv:2507.11344 — 공정성 보상모델로 추론단계 점수화.

### VLM 편향 벤치마크 (과제 맥락)
- **VisBias** (2025) arXiv:2503.07575 · **VLMs are Biased** (ICLR 2026) · **Vignette** arXiv:2505.22897 — VLM 사회편향 측정. (우리는 VLM이되 placeholder 이미지라 텍스트편향 중심.)

### 정직성 메모
모든 인용은 2026-06-16 WebSearch로 제목·저자·arXiv번호 확인함. 구현은 Unsloth GRPO 템플릿(회원님 train_grpo.ipynb) 기반이며, 위 논문들은 방법론적 근거(GRPO 식·콜드스타트·dynamic sampling·bias score 정의)를 제공한다.
