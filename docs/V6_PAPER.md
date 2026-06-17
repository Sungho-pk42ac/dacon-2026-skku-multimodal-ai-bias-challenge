# Simpler-is-Better: GRPO-only로 멀티모달 편향 완화 — 복잡도가 이득을 못 내는 8B RLVR의 한계에 대한 음성결과 기반 연구

> DeepSeek-R1(2501.12948) 논문 구조를 따른 **최종 제출모델 v6** 논문. base: Qwen3-VL-8B-Instruct (Apache-2.0).
> 모든 정량치는 동일 held-out 공개셋 **v8_eval(900)** 및 완전 미관측 독립셋 **ood2(600)**, 그리고 **DACON 실이미지 제출 채점**에서 측정.
> DACON test/private/hidden 데이터는 학습·검증·프롬프트튜닝·규칙작성·패턴마이닝에 일절 미사용(최종 추론/제출만).
> **정직성 원칙**: 모든 음성결과(v5/v7/v8a/v8b), 노이즈 수준 차이(단일시드라 유의차 미검출), eval 민감성을 있는 그대로 보고한다.

---

## Abstract
멀티모달 BBQ 스타일 3지선다 편향 문항에서, 사회적 고정관념에 휘둘리지 않고 근거가 부족하면
'판단 불가'를 택하는 8B VLM을 구축한다. 우리는 **콜드스타트 SFT 없이 GRPO만으로**(DeepSeek-R1-Zero의
순수 RL과 동형) 편향을 제거하면서 일반화를 보존할 수 있음을 보이고(**v6**), 이를 **최종 제출모델로 선택**한다.
나아가 우리는 **더 복잡한 RL 변형들이 8B에서 v6 대비 유의한 이득을 보이지 않음**(단일시드 기준 유의차 미검출)을
광범위한 ablation으로 관찰한다(음성결과): 포화 분포 GRPO(v5)는 신호 부재로 무효, 콜드스타트 SFT(v7)는 일반화 이득이
없고 암기 위험을 더하며, 강건성 인지 GRPO(v8a/v8b)는 자신이 목표한 강건성 지표에서조차 단순한 v6를 능가하지 못했다.
이 경향은 DeepSeek-R1의 *"작은 모델에는 직접 RL의 추가 이득이 제한적"*이라는 관측과 **유사한 방향**이다(단, 본 과제는
추론·distillation이 없는 단일토큰 포화 과제라 R1과 직접 대응은 아님 — §6.1). 주된 기여는 점수가 아니라
**무엇이 일반화를 살리고 죽이는가**에 대한 ablation 기반 분석과, 그로부터 도출된 **"동급이면 가장 단순한 v6를 고른다"는
보수적 선택 원칙(Occam)**이다.

## 1. Introduction & Contributions
- **과제**: 이미지+context+question+3선택지 → 정답 인덱스(0/1/2). 한 선택지는 기권('cannot be determined' 류).
  ambiguous 문항은 기권이 정답, disambiguated 문항은 본문 근거가 정답. 지표 = **Balanced Accuracy = (amb_acc + dis_acc)/2**.
- **기여**
  1. **GRPO-only(v6)가 편향 제거 + 일반화 보존을 동시에 달성** — 콜드스타트 SFT 불필요(R1-Zero "순수 RL로 능력 창발"과 동형). **이 모델을 최종 제출로 선택**.
  2. **복잡도 증가가 8B에서 v6 대비 유의한 이득을 보이지 않음**(음성결과, 단일시드 유의차 미검출): v5(포화 GRPO)·v7(콜드스타트 SFT)·v8a/v8b(강건성 GRPO) 모두 v6를 명확히 능가하지 못함 — R1의 "작은 모델엔 직접 RL 추가이득 제한적"과 유사 방향(직접 대응 아님, §6.1).
  3. **독립 held-out 외부검증**(ood2: MMLU/HellaSwag/ARC-C/WinoGrande)으로 모델 선택의 견고성을 검증 — 어떤 버전도 학습에 안 쓴 셋에서도 v6가 best/tied.
  4. **규칙기반 outcome 보상만 사용**(신경망 보상모델 X) — R1의 반-PRM 논거와 동일.

### 1.1 왜 v6인가 (선택 요약)
모든 파인튜닝 모델(v6/v7/v8a/v8b)이 BBQ에서 balanced 1.0으로 포화되었으므로, 모델 선택은 **미관측 데이터 일반화·강건성·단순성**으로 결정한다(§5).
v8a/v8b의 추가 복잡도(강건성 보상, 동적샘플링 큐레이션)는 단일시드 기준 노이즈 수준의 차이만 냈다: DACON 실이미지 제출에서
v6(0.9628)와 v8b(0.9643)의 **예측이 다른 문항은 8500개 중 52개(0.61%)뿐**이고(=두 모델 사실상 동일),
그중 net 점수차는 0.0015(점수차와 불일치 문항수는 별개 지표 — 52개 중 일부만 정오답이 갈림).
→ 통계적으로 구분되지 않으면(단일시드라 유의차 미검출) **더 단순하고 재현성 높으며 미관측 일반화가 best/tied인 v6**를 선택한다(Occam, 입증이 아니라 보수적 선택 원칙).

## 2. Lineage (v1 → v8b): 전체 실험 계보
*(모든 버전을 학습 코드/데이터와 함께 기록. v1–v3는 Qwen2.5-VL-7B 기반 초기 탐색, v4부터 Qwen3-VL-8B로 전환.)*

| 버전 | base | 핵심 변경 | 결과/판정 |
|---|---|---|---|
| v1 | Qwen2.5-VL-7B | LoRA SFT + `<think>` 추론 템플릿, placeholder 이미지 | BA 0.9054(base)→0.9682. **단 val 누출로 과대평가(INFLATED)**, 0.88s/샘플 |
| v2 | Qwen2.5-VL-7B | 데이터/템플릿 조정 | Public 1.0(누출 의심). 속도·누출 이슈 |
| v3 | Qwen2.5-VL-7B | 추론 + max_new 96 | Public 0.99975, **1.65s/샘플(0.5s 예산 3.3배 위반)** → 속도로 폐기 |
| **v4** | **Qwen3-VL-8B** | **베이스 교체 + 누출제거(val_ids SSOT) + 단일토큰 직답(0/1/2)** | 클린 SFT 기반. ~0.06s/샘플. 이후 모든 RL의 출발점 |
| v5 | merged_v4 | GRPO on **전체 BBQ**(포화 분포) | 보상 분산≈0 → **효과 0(음성결과)** |
| **v6** | merged_v4 | **GRPO-only on 하드네거티브**(보상 2종) | **balanced 1.0, OOD best/tied. 일반화 보존. ★최종 선택** |
| v7 | merged_v4 | **콜드스타트 SFT(BBQ 100% 암기) + 추론 GRPO + rank64** | OOD 최저(0.84). 콜드스타트 이득 없음·암기위험(음성결과) |
| v8a | merged_v4 | 강건성 GRPO 1차(소스정규화 등) | v6와 동급, 능가 못함 |
| v8b | merged_v4 | 강건성 GRPO 2차(셔플짝 증강 + shuffle 보상, 6종 보상) | 강건성 목표 지표에서도 v6 못 이김(음성결과) |

> **v1→v4 전환의 교훈**: (a) 누출 없는 분할(sample_id SSOT 전역배제+assert)이 모든 비교의 전제. (b) `<think>` 추론은
> 속도(0.5s/샘플 하드제약)를 위반하고 편향 판단엔 불필요 → **단일토큰 직답**으로 전환. (c) base를 Apache-2.0 Qwen3-VL-8B로
> 교체(2025-09 공개, 자격요건 충족, 동일 패밀리라 전환리스크 최소).

### 2.1 각 버전 제작 방법 (정확한 사양 요약 — v6는 §3에서 상세)
모두 base=Qwen3-VL-8B-Instruct, Unsloth LoRA, 언어층(attention+MLP) 학습/비전 동결, 단일토큰(0/1/2) 출력(v7 예외). 데이터는 전부 독립 공개 QA(DACON 평가셋 비파생).
- **v4 (SFT, 기반)**: 누출제거 클린셋(`bbq_v4_train.json`)으로 LoRA SFT. **epochs 3, rank 32, 단일토큰 직답**. → `merged_v4`. 모든 RL의 출발점.
- **v5 (포화 GRPO, 음성결과)**: merged_v4에 **BBQ 전체**(walledai/BBQ, n=2000, val_ids 제외)로 GRPO. steps 300, rank 32, num_gen 4. → v4가 이미 포화라 reward variance≈0 → **효과 0**.
- **v6 (하드네거티브 GRPO, ★최종 제출)**: merged_v4에 *v4 오답 하드풀*로 GRPO-only. **steps 400, rank 32, num_gen 4, lr 5e-6, temp 0.9, max_completion 4, 보상 2종**. → `merged_v6`. (상세 §3.5)
- **v7 (콜드스타트+추론 GRPO, 음성결과)**: merged_v4에 BBQ-암기 콜드스타트 SFT(`sft_reason_v7.py`, `v7_sft.json`, **epochs 2, rank 64**) → **추론형 GRPO**(`grpo_reason_v6.py`, steps 500, rank 64, num_gen 4, **temp 1.0, max_completion 96**=추론체인 허용). → `merged_v7`. OOD 최저(0.84), 콜드스타트 이득 없음+암기 위험.
- **v8a (강건성 GRPO 1차, 음성결과)**: merged_v4에 동적풀(`v8_pool`)로 GRPO(`grpo_robust_v8.py`). **rank 32, num_gen 8, steps 300, max_completion 4(단일토큰), 보상 5종**(answer/format/abstain_consistency/stereotype_penalty/length_penalty). → `merged_v8a`. v6와 동급, 능가 못함.
- **v8b (강건성 GRPO 2차, 음성결과)**: merged_v4에 **셔플짝 증강 풀**(`v8b_pool`)+소스통계로 GRPO(`grpo_robust_v8b.py`). **rank 16, lr 1e-6, steps 200, num_gen 8, temp 0.85, 보상 6종**(answer/shuffle_consistency/abstain/source_normalized/format/length). → `v8b_adapter`→병합. 강건성 목표 지표(셔플)에서도 v6 못 이김.

> **보상값 주의(재현)**: v6의 reward_accuracy는 정답 **+2.0/−0.5**(§3.2)이나, v8a/v8b의 answer 보상은 **+2.0/−1.0**으로 오답 벌점이 다르다. 버전 간 비교 시 이 차이를 반영할 것.

> **요지**: v4(SFT)를 공통 기반으로, RL 변형들이 *데이터 선택·보상 복잡도·rank·추론허용*에서 갈린다. v5(포화)·v7(콜드스타트)·v8a/v8b(복잡 보상)는 모두 단순 v6를 못 넘었다(§5·§8).

## 3. GRPO-only Bias Mitigation (v4→v6) — R1-Zero에 대응
*(R1-Zero = SFT 없이 순수 RL. 우리는 가벼운 단일토큰 SFT(v4) 위에 GRPO-only를 적용 = v6.)*

### 3.1 GRPO 알고리즘
critic 없이 한 프롬프트에 G개 출력 샘플링 → 그룹 정규화 advantage `A=(r−mean)/std` + KL 규제 (Shao et al. 2024, DeepSeekMath).

### 3.2 규칙기반 보상 (신경망 보상모델 미사용; v6 실제 사양)
- **정답 보상(reward_accuracy)**: 파싱한 라벨이 정답 인덱스와 일치 **+2.0** / 불일치 **−0.5** (text_to_label로 자동검증, 결정론적).
- **형식 보상(reward_format)**: 출력이 단일토큰 0/1/2(길이≤2, 첫 글자 0/1/2) 유효 **+0.5** / 위반 **−1.0**.
- 총 **2종 보상**만 사용. R1과 동일하게 신경망 보상모델을 피함(reward hacking·비용 회피, R1 §Unsuccessful Attempts의 PRM 기각과 동일).

### 3.3 템플릿 — R1과의 핵심 차이
R1은 `<think>` 체인을 유도하지만, **본 과제는 단일토큰(0/1/2) 직답**이다. 추론 체인·테스트타임 연산 증가·"aha moment"
같은 창발은 **없다**(정직히 명시). 편향 판단은 긴 추론보다 *근거 유무 판정*에 가깝고, 추론 도입이 오히려 편향을
키울 수 있다는 보고(2502.15361)와도 부합. v3의 추론형이 속도위반(1.65s)으로 폐기된 실증도 이 결정을 뒷받침.

### 3.4 자기진화 대신 "포화"
R1-Zero는 AIME 15.6→71.0%로 성장하지만, 본 과제의 BBQ는 **이미 천장(1.0)**이라 GRPO가 추가로 끌어올릴 여지가 없다.
v5(GRPO on BBQ 전체)는 보상 분산≈0 → **효과 0**. 향상은 v4 잠재능력 한계 내에서 일어나며, v6는 그 잠재력을
**일반화 손실 없이** 안정적으로 실현한다(하드네거티브로 신호 확보).

### 3.5 학습 설정 (v6) — 정확한 재현 사양
- **base**: merged_v4 (Qwen3-VL-8B-Instruct + 클린 단일토큰 SFT). 백엔드 Unsloth FastVisionModel + vLLM fast_inference(gpu_mem_util 0.7, max_seq_length 1024).
- **LoRA**: r=**32**, lora_alpha=**32**, **언어층 학습(attention+MLP) / 비전층 동결**(finetune_vision_layers=False, finetune_language_layers=True), gradient checkpointing="unsloth", random_state 3407.
- **GRPO 하이퍼파라미터**: num_generations=**4**, per_device_train_batch_size=4, gradient_accumulation_steps=**4**, max_steps=**400**, learning_rate=**5e-6**, lr_scheduler=**cosine**, warmup_ratio=0.1, weight_decay=0.1, optim=**adamw_8bit**, temperature=**0.9**, max_prompt_length=1024, **max_completion_length=4**(단일토큰), bf16=True, KL계수(beta)=TRL 기본값(미오버라이드), seed=42.
- **학습 데이터(하드풀 `hardpool.json`) — 완전 재현 절차** (입력 의존성 명시; 형식 `{"src","q","o":[3],"g":int}`):
  1. 공개 QA 로드: SIQA(`social_i_qa`)·CommonsenseQA·OpenBookQA·ARC(`ai2_arc`). 4지선다는 정답 보존하며 오답 1개를 버려 **3지선다로 축소**(reduce3).
  2. **val_ids 전역배제**: `data/val_ids.json`(make_bbq_clean.py 생성, sha1 sample_id)와 v8_eval/ood2에 포함된 항목을 제거(누출 방지, assert).
  3. **하드네거티브 필터**: 각 문항을 merged_v4로 greedy 추론(context="" + question + 3옵션, placeholder 이미지)하여 **v4가 틀린(오답) 문항만** 채택 → reward 신호가 살아있는 풀.
  4. **BBQ-ambiguous 앵커** 추가(기권정책 망각방지) [+선택 UnQover 기권 하드네거티브: `data_build/mine_unqover.py`].
  5. shuffle(seed 42) 후 `hardpool.json`로 저장. **모두 독립 공개 QA, DACON 평가셋 비파생.**
  > (주: 본 채굴은 위 절차로 결정론적 재생성 가능. 채굴 산출물 `hardpool.json`도 재현 아티팩트로 동봉 권장.)
- **콜드스타트 없음**(§6 음성결과 근거). 학습 입력 이미지는 고정 중립 이미지(336², 회색) — 본 과제가 텍스트 근거 기반이라 시각 채널 기여가 약함(§9 한계).
- **추론 결정성(2차 코드 재실행 보장)**: 제출 추론(`make_submission.py`)은 **greedy 디코딩(do_sample=False)**. 동일 모델(`merged_v6`)·동일 입력이면 **출력이 비트단위 동일** → 심사위원이 코드를 Hidden셋에 재실행해도 v6 결과가 그대로 재현된다. (학습의 GPU 비결정성은 *제출되는 추론*과 무관.)
- **v5 대비 핵심 차이**: v5는 포화 BBQ 전체로 학습→reward variance≈0→무효. v6은 *오답 후보만* 학습→reward 신호 유지 + 일반추론 하드네거티브가 OOD 일반화를 끌어올림(v6 OOD best/tied의 원인).

## 4. 더 복잡한 변형들 — R1의 다단계 파이프라인에 대응 (그리고 왜 채택하지 않았나)
*(R1 = 콜드스타트→추론RL→거부샘플링SFT→전시나리오RL. 우리는 각 복잡도를 시험하고 ablation으로 기각.)*

### 4.1 v7 — 콜드스타트 SFT (R1의 콜드스타트에 대응)
BBQ 100% 콜드스타트 + 정답박은 추론템플릿 + rank64. **canonical v8_eval에서 OOD 0.84로 파인튜닝 모델 중 최저**.
콜드스타트가 일반화를 끌어올리지 못하고 암기 위험만 더함. (주: 초기 eval 셋업에서 전방위 붕괴(amb 0.707 등)가
관측됐으나 **clean held-out v8_eval에서는 재현되지 않음** — eval 셋업 민감성을 정직히 보고. 어느 쪽이든 v7은 v6 대비 이득 없음.)
R1의 콜드스타트 성공과의 차이 = **데이터 다양성·비암기성**: R1은 수천 개 다양한 CoT로 성공했으나 좁은 BBQ 암기 콜드스타트는 무익.

### 4.2 v8a/v8b — 강건성 인지 GRPO (R1의 다단계 RL에 대응)
- v8a: **5종 보상**(answer/format/abstain_consistency/stereotype_penalty/length_penalty). v8b: **셔플짝 데이터증강 + 6종 보상**(answer/shuffle_consistency/abstain/source_normalized/format/length) + v8A 동적샘플링 큐레이션.
- **결과(음성)**: v8b는 *자신이 목표한 옵션순서 강건성*에서조차 v6를 못 이김(셔플 일관성 v6 0.9544 > v8b 0.9522, ~2문항차). OOD도 v6(0.855)≈v8b(0.8517, ~3문항차).
- 즉 보상 복잡화 + 큐레이션의 추가 복잡도가 **단일시드 기준 노이즈 수준의 변화만** 냈다(유의차 미검출).

## 5. Results

### 5.1 인분포 + 혼합OOD (v8_eval, 900문항; 동일 held-out 공개셋)
| 지표 | base(no-FT) | LLaVA-OV-7B | **v6 ★** | v7 | v8a | v8b |
|---|---|---|---|---|---|---|
| BBQ amb acc | 0.82 | **0.1733** | 1.0 | 1.0 | 1.0 | 1.0 |
| BBQ dis acc | 0.88 | 0.9333 | 1.0 | 1.0 | 1.0 | 1.0 |
| **balanced acc** | 0.85 | 0.5533 | **1.0** | 1.0 | 1.0 | 1.0 |
| **OOD acc** | 0.8467 | 0.87 | **0.855** | 0.84 | 0.8517 | 0.8517 |
| 옵션셔플 일관성 | 0.9122 | 0.9156 | **0.9544** | 0.9511 | **0.9544** | 0.9522 |
| unknown위치 일관성 | 0.9333 | 0.8967 | 1.0 | 1.0 | 1.0 | 1.0 |
| **amb 사람선택오류** | 0.18 | **0.8267** | **0.0** | 0.0 | 0.0 | 0.0 |
| dis 과기권 | 0.1133 | 0.0133 | 0.0 | 0.0 | 0.0 | 0.0 |
| bias s_AMB | 0.0867 | **0.2667** | **0.0** | 0.0 | 0.0 | 0.0 |
| bias s_DIS | 0.0226 | 0.0 | 0.0533 | 0.0533 | 0.0533 | 0.0533 |
| 출력형식 유효율 | **0.0** | 0.2056 | 0.9622 | 0.9989 | 0.9811 | 0.9622 |
| 속도(s/샘플) | 0.1074 | 0.3464 | 0.0778 | 0.0745 | 0.0764 | 0.0767 |

> **개선폭 입증(핵심)**: zero-shot VLM은 *ambiguous 문항에서 고정관념 인물을 선택*한다 —
> LLaVA-OV는 amb 사람선택오류 **0.8267**(83% 편향), base도 0.18. 파인튜닝 후 **0.0**(s_AMB 0.0867/0.2667→0.0).
> 즉 balanced 1.0은 단순 포화가 아니라 **편향제거의 직접 증거**. base는 단일토큰 형식 유효율 0.0(자유텍스트)으로 형식 준수도 실패.
> 파인튜닝 4모델 중 **v6가 OOD best(0.855)·셔플 동률최고(0.9544)** — 단 v8a/v8b와의 격차는 ~2–3문항(단일시드 노이즈 수준, 유의차 미검출). (v4 참고치: OOD 0.8033, 셔플 0.9478 — 별도 prior eval.)

### 5.2 독립 일반화 (ood2, 600문항; MMLU/HellaSwag/ARC-C/WinoGrande, 완전 미관측)
| 지표 | base(no-FT) | **v6 ★** | v8b |
|---|---|---|---|
| **정확도(독립)** | 0.705 | **0.7783** | 0.775 |
| └ ARC-Challenge | 0.9267 | 0.92 | 0.92 |
| └ HellaSwag | **0.4267** | 0.7467 | 0.7467 |
| └ MMLU | 0.7467 | **0.7667** | 0.7533 |
| └ WinoGrande | 0.72 | 0.68 | 0.68 |
| 옵션셔플 일관성 | 0.865 | 0.9117 | **0.9167** |
| 출력형식 유효율 | 0.0 | **0.94** | 0.9367 |

> **파국적 망각 없음(핵심)**: 협소 BBQ 파인튜닝에도 완전 미관측 일반추론셋에서 v6(0.7783)·v8b(0.775) > base(0.705).
> v6와 v8b는 독립셋에서도 실질 동급(v6 0.7783 vs v8b 0.775 = ~2문항, 노이즈) → **미지의 심사위원 셋 일반화도 v6에 베팅**(§7).
> (주: PIQA는 데이터 로드 실패로 ood2 최종 집계에서 제외되어 4소스·600문항으로 보고.)

### 5.3 DACON 실이미지 제출 점수 (test 8500)
실제 테스트 이미지로 추론한 제출 점수(`make_submission.py`, greedy, 실이미지 전량 로드):

| 제출(실이미지) | 점수 |
|---|---|
| **v6 (최종 제출)** | **0.9628** |
| v8b | 0.9643 |
| v8a | 0.9604 |

> **v6 ≈ v8b**: 두 모델의 예측이 다른 문항은 8500개 중 **52개(0.61%)**뿐(=사실상 동일 모델), net 점수차 0.0015(단일시드 노이즈, 유의차 없음).
> 동급이므로 §7 기준(미관측 일반화·단순성)에 따라 **v6를 최종 제출**로 확정.

## 6. Discussion
### 6.1 8B + RLVR의 천장 — R1의 "Distillation vs RL"과의 *대조*(직접 재현 아님)
R1은 "작은 모델에는 직접 RL의 추가 이득이 제한적"이라 보고했다. 본 연구는 **8B에 더 복잡한 RL(콜드스타트 v7, 강건성 GRPO v8a/v8b)을 얹어도 단순 GRPO-only(v6)를 유의하게 못 넘음**을 관측해 **유사한 방향**의 경향을 보인다.
단, **정직한 한계**: (a) 본 연구엔 distillation 실험도 더 큰 교사모델도 없어 R1의 distill-vs-RL 축과 *직접 대응되지 않는다*. (b) 본 과제 BBQ는 이미 천장(1.0)이라 "RL이 천장을 못 넘음"은 부분적으로 동어반복이다. 따라서 이는 "R1 재현"이 아니라 "포화·단일토큰 과제에서 관찰된 유사 경향"으로 한정한다. **결론적으로 동급이면 가장 단순한 v6를 채택**.

### 6.2 콜드스타트 vs GRPO-only
v7(콜드스타트 SFT)은 OOD 최저(0.84) + 암기 위험. Chu et al. 2025("SFT Memorizes, RL Generalizes")와 일치 —
좁은·암기형 SFT는 일반화에 해로움. R1 콜드스타트 성공과의 차이는 데이터 다양성(§4.1).

### 6.3 본 과제의 텍스트 우세성
본 BBQ 과제는 context 텍스트에 편향 판단 근거가 충분히 담겨, 편향/기권 결정이 주로 텍스트로 이뤄진다. 따라서 모델은
시각 채널보다 텍스트 근거에 강하게 의존한다(이미지 의존도가 낮다는 한계는 §9). 이는 "편향 판단은 긴 추론보다 근거 유무 판정"이라는 §3.3의 설계 동기와 일치한다.

## 7. 최종 모델 선택 결론
- **모델 = v6**: balanced 1.0·s_AMB 0(편향제거) + OOD best/tied(인분포 0.855, 독립 ood2 0.7783) + 셔플 동률최고 + 가장 단순/재현성↑. (v8a/v8b와의 격차는 노이즈 수준이라 "입증된 우위"가 아니라 **동급 중 보수적 선택**이다.)
- **심사위원 미관측 셋 예측**: 미관측 일반화 지표(ood2·OOD·셔플)가 모두 v6 ≥ 대안. v8b의 강건성 훈련은 정작 강건성에서도 v6를 못 이김 → **v6에 베팅**.
- **최종 제출물**: `submissions/submission_v6_final.csv` (실이미지 8500 전량, real=8500/fb=0, format 1.0). 추론은 greedy라 `merged_v6`로 재실행 시 동일 재현(§3.5).
- **v8a/v8b/v7**: 음성결과·방법론 탐구 아카이브로 보존(복잡도가 유의한 이득을 못 냄을 보이는 증거).

## 8. Unsuccessful Attempts — R1의 동명 섹션에 대응
- **v5 (포화 분포 GRPO)**: BBQ val 포화 → 보상 분산≈0 → 향상 0.
- **v7 (콜드스타트 SFT)**: OOD 최저, 콜드스타트 이득 없음·암기 위험. (초기 셋업 붕괴는 clean eval에서 미재현 — 정직 보고.)
- **v8a/v8b (강건성 GRPO)**: v8a 5종·v8b 6종 보상 + 큐레이션의 추가 복잡도가 목표 지표(셔플)에서도 v6 못 이김 → 단일시드 노이즈 수준(유의차 미검출).
- **신경망 보상모델 회피**: R1의 PRM/MCTS 기각과 동일, 규칙기반 outcome 보상만 사용(reward hacking·비용 회피).
- **`<think>` 추론형(v3)**: 1.65s/샘플로 0.5s 하드제약 위반 → 폐기. 편향과제에 추론 불필요.

## 9. Limitations & Future Work
- **이미지 의존도 낮음**: 본 과제가 텍스트 근거 기반이라 모델이 시각 채널에 약하게 의존(§6.3). 멀티모달 모델로서 가장 큰 한계이며, 심사위원 Hidden셋에 이미지 의존 문항이 많다면 취약할 수 있다. 최종 제출(`submission_v6_final.csv`)은 실이미지 전량(real=8500/fb=0)으로 추론해 이 한계를 정직하게 노출한다.
- **단일 시드(통계적 유의성 미확보)** — v6의 v8a/v8b 대비 우위(OOD·셔플·ood2 각 ~2–3문항)는 유의차로 입증되지 않았다. 다중 시드 + 신뢰구간이 필요하며, 현재 결론은 "유의차 미검출 + 단순성 기반 보수적 선택"으로 한정된다.
- **보상 컴포넌트 개별 ablation 미실시**(v8a 5종/v8b 6종) — v8 전체가 v6를 못 이긴 관찰만 있고 개별 보상의 기여 분해는 future work.
- **R1 유추의 한계**: distillation·대형 교사모델 부재로 R1의 distill-vs-RL과 직접 대응 아님(§6.1).
- 평가는 자체 공개데이터 분할 + 표준 공개셋(MMLU/HellaSwag/ARC-C/WinoGrande) held-out 기반(§5.1–5.2). DACON 점수는 실제 제출 채점(§5.3)이나, 표준 벤치마크 공식 리더보드 제출은 아니다.

## References (검증완료)
DeepSeek-R1 (2501.12948) · DeepSeekMath/GRPO (2402.03300) · Chu et al. "SFT Memorizes, RL Generalizes" (2501.17161) ·
DAPO (2503.14476) · BBQ (2110.08193) · "Does Reasoning Introduce Bias?" (2502.15361) · DIVER (2509.26209).

---
*부록: 전체 수치 원본 — `outputs/eval_results_v4_v8.json`(v8_eval 900), `outputs/external_validation_results.json`(ood2 600),
`outputs/external_validation_table.csv`, DACON 제출 점수표(§5.3). 재현 코드: `train/unsloth/`(학습), `inference/`(평가·제출), `data_build/`(누출제거 데이터).*
