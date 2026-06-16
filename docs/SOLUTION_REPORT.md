# 멀티모달 편향 질의응답에서 기권 기반 공정성 모델 — 기술 보고서

> 2026 성균관대학교 멀티모달 AI Bias 챌린지 (대회 236722) · 솔로 · 박성호
> 본 문서 = **산출물 ①(Private 복원 코드/모델/외부데이터 설명서)** + **발표 PDF의 기술 백본**.
> **정직성 원칙**: 방법·하이퍼파라미터·환경·실패이력·참고문헌은 *실제 값*으로 기재. 실험 결과(BA·속도·강건성)와 W&B/스크린샷은 `〖CAPTURE〗` 표식 자리에 **학습·평가 완료 후 실측 산출물을 삽입**하며 임의 수치 기재를 금한다.

---

## 초록 (Abstract)
멀티모달 BBQ형 3지선다 과제에서, 근거가 충분하면 정답을 고르고 부족하면 *기권("cannot be determined")* 하는 모델을 개발한다. 채점지표 Balanced Accuracy=(Acc_ambiguous+Acc_disambiguated)/2 의 절반이 기권 품질에 달려 있다. 본 연구의 핵심 기여는 **(i)** 자체 검증셋의 누출을 코드 수준에서 적발·제거하여 *정직한 BA* 측정 체계를 세운 점, **(ii)** 옵션 순서 셔플로 다지선다 위치편향을 제거한 점, **(iii)** 단일토큰 직답으로 추론속도 규칙(0.5초/샘플)을 충족한 점, **(iv)** GRPO로 Balanced Accuracy를 직접 보상 최적화한 점이다. 모든 학습은 공개 BBQ만 사용하며(외부증강 0), Private/Hidden 일반화를 1차 목표로 한다.

## 1. 문제 정의와 평가
- **입력**: 이미지 + context + question + 3개 선택지. **출력**: 정답 인덱스 0/1/2.
- **지표**: Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2.
  - *ambiguous*: 근거 부족 → 정답은 "cannot be determined"류 기권 옵션.
  - *disambiguated*: 근거 충분 → 특정 옵션이 정답.
- **제약(컴플라이언스)**: 학습데이터 미제공(예시 1건) · 추론 ≤0.5초/샘플(A6000 기준) · 오프라인(API 0) · 베이스 모델 2026-06-01 이전 공개 · 최종답은 LLM 생성물 · 학습/추론 코드 분리 · 외부데이터 평가셋 모방 금지(Data Leakage).
- **운영진 채점 철학(2026-06 공지)**: Public은 참고지표일 뿐, *신빙성 있는 자체 검증 + 다양한 입력/상황 강건성 + Public/Private 간극을 고려한 일반화* 를 요구. 본 솔루션은 이 철학을 설계 원칙으로 삼는다(§3.1, §5.3).

## 2. 관련 연구 (Related Work)
| 차용한 아이디어 | 문헌 | 본 연구에서의 사용 |
|---|---|---|
| 편향 벤치마크, ambiguous/disambiguated 2단 평가 | Parrish et al., *BBQ: A Hand-Built Bias Benchmark for QA*, Findings of ACL 2022 (arXiv:2110.08193) | 학습/검증 데이터의 원천 및 평가 프레임 |
| 추론(CoT)을 학습으로 내재화·증류 | DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*, 2025 (arXiv:2501.12948) | v2의 진짜 CoT 증류 SFT 계보 |
| 다지선다 **선택편향/위치편향** | Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors*, 2023 (arXiv:2309.03882) | §3.1 옵션 셔플의 직접 근거 |
| 옵션 순서 민감도 | Pezeshkpour & Hruschka, *Sensitivity to the Order of Options in MCQ*, 2023 (arXiv:2308.11483) | 위치편향 제거 보강 근거 |
| **기권 학습 = 일반화되는 meta-skill** | Zhang et al., *R-Tuning: Instructing LLMs to Say 'I Don't Know'*, 2023 (arXiv:2311.09677) | 기권 SFT가 OOD로 일반화된다는 근거 |
| **GRPO**(critic 없는 RL, 그룹 상대 보상) | Shao et al., *DeepSeekMath*, 2024 (arXiv:2402.03300) | §3.3 v5의 BA 직접 최적화 |

## 3. 방법 (Method)

### 3.1 데이터 전처리 — `data_build/make_bbq_clean.py`
원천: 공개 `walledai/BBQ` (외부증강·합성 0). 처리 순서:
1. **품질필터**: 선택지 3개 · "cannot be determined"류 기권 옵션 정확히 1개(`UNKNOWN_PATTERNS` 15종 매칭) · 라벨∈{0,1,2}. 위반 시 폐기.
2. **지문 해시**: `sample_id = sha1(context|question|options)[:16]` (중복제거 + 검증셋 SSOT). `scenario_id = sha1(question|sorted(options))[:16]` (같은 상황의 amb/dis 묶음).
3. **시나리오 단위 train/val 분리(atomic)**: 카테고리별로 시나리오를 셔플, 카테고리당 `val_per_cat=120`개를 검증셋으로 통째 이동. **`assert` 로 시나리오 누출=0 보장.** `val_ids.json` 로 전역 배제.
4. **옵션 셔플 + 라벨 재매핑**: `sample_id` 기반 결정적 셔플(재현 가능)로 [0,1,2] 순서를 섞고 정답 라벨을 새 위치로 매핑. → 정답 위치 0/1/2 분포 균등화(위치편향 제거; Zheng 2023).
5. **단일토큰 타깃**: assistant 출력 = `"0"|"1"|"2"` (sharegpt 포맷, 시스템=공정성·기권 지시 프롬프트).
- 출력물: `val_ids.json`(SSOT) · `bbq_val_clean.jsonl`(검증) · `bbq_v4_train.json`(학습).
- **데이터 통계(실측)**: 고유문항 **58,414**(불량 0) → TRAIN **57,094**(amb 28,546 / dis 28,548) · VAL **1,320**(amb 660 / dis 660, 완전균형). 셔플 후 정답 위치 분포 **0=19,196 · 1=18,922 · 2=18,976**(거의 균등 → 위치편향 제거 실증). **시나리오 누출 = 0** (assert 통과). 학습 표본 57,094 / 유효배치 16 × 3에폭 ≈ **10,705 step**(실측 진행바 10,707과 일치).

### 3.2 베이스 모델·SFT — `train/unsloth/sft_unsloth.py`
- 베이스: **Qwen/Qwen3-VL-8B-Instruct** (2025-09 공개 = 6/1 이전 ✓, Apache-2.0 ✓), Unsloth `FastVisionModel`.
- LoRA: **비전타워 동결**(placeholder 이미지라 시각정보 없음), 언어·어텐션·MLP 층만 학습. `r=32, lora_alpha=32, lora_dropout=0.0, bias=none`. `max_seq_length=2048`, gradient checkpointing.
- SFTConfig(실제 값): `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (유효배치 16), `learning_rate=2e-4`, `lr_scheduler=cosine`, `warmup_ratio=0.1`, `num_train_epochs=3`, `bf16=True`, `optim=adamw_8bit`, `weight_decay=0.01`, `seed=42`.
- 총 스텝 = 10,707 (≈ 학습표본/16 × 3에폭). 저장: `save_pretrained_merged(merged_16bit)` → `outputs/merged_v4`.

### 3.3 v5 — GRPO (`train/unsloth/grpo_unsloth.py`)
- merged_v4 위에서 GRPO(critic-free, 그룹 상대 보상; Shao 2024). 보상 = **정답 일치 + 포맷(0/1/2) = Balanced Accuracy 직접 최적화**.
- `val_ids.json` 배제 + 동일 옵션 셔플 → v4와 완전 동일 전처리(일관성). `fast_inference=True`(vLLM)로 rollout 가속(학습시간 단축, 최종 추론속도와 무관).

### 3.4 추론·파싱 — `prompts.py:text_to_label`, `inference/make_submission.py`
- 추론: `max_new_tokens=4`, greedy. 생성 텍스트 → `text_to_label`로 0/1/2 정규화(선행숫자→텍스트일치→문자겹침). **최종답은 LLM 생성물**(룰 매핑 아님).

## 4. 실험 설정 (Experimental Setup) — 실측

### 4.1 환경 (2 venv 분리)
| 구분 | 값 |
|---|---|
| OS / Kernel | Ubuntu 22.04.5 LTS / 5.15.0-139-generic |
| GPU / Driver / CUDA | NVIDIA RTX A6000 48GB / 550.127.08 / CUDA 12.4 |
| Python | 3.11.10 (양쪽 venv) |
| **학습 `unsloth_env`** | unsloth 2026.6.7 · unsloth_zoo 2026.6.5 · torch 2.10.0 · transformers 4.57.6 · trl 0.24.0 · peft 0.19.1 · vllm 0.18.0 · xformers 0.0.35 · triton 3.6.0 · bitsandbytes 0.49.2 · accelerate 1.14.0 · datasets 4.3.0 |
| **추론/데이터 `lf311`** | torch 2.6.0+cu124 · torchvision 0.21.0+cu124 · transformers 5.6.0 · accelerate 1.14.0 · peft 0.19.1 · datasets 5.0.0 · qwen-vl-utils 0.0.14 · pillow 12.2.0 · pandas 3.0.3 · numpy 2.4.4 · huggingface-hub 1.19.0 |
> 추론(제출) 환경은 대회 기준환경(A6000 48GB)과 일치. 학습은 unsloth 전용 의존성(torch 2.10)으로 분리.

### 4.2 학습 진행 〖CAPTURE: W&B〗
- W&B project `dacon-bias-challenge`, run `sft-v4-unsloth-qwen3vl` (online). **삽입할 그림**: ① train/loss 곡선(단일토큰 특성상 빠른 초기 하락 후 저평탄지대) ② learning_rate(cosine) ③ grad_norm 안정성. → §6 실패분석의 "정직한 학습" 논거로 사용.

### 4.3 실행 방법 · 단계별 소요시간 (A6000 48GB 기준, **v5 제외**)
> 전체 자동: `setsid bash infra/pod_master_unsloth.sh >…/master_unsloth.log 2>&1 </dev/null &` (데이터→v4 SFT→평가→제출→v5→선택을 무인 연속 실행). 아래는 **v4 경로 수동 재현** 명령과 소요시간.

| 단계 | 명령 | venv | 소요시간 | 산출물 |
|---|---|---|---|---|
| 0. 환경 구축 | uv venv ×2 (torch·transformers·unsloth·vllm) | — | **~15–20분** (실측) | `/workspace/lf311`, `/workspace/unsloth_env` |
| 1. 데이터 생성 | `python data_build/make_bbq_clean.py` | lf311 | **~3–5분** (실측) | `val_ids.json`·`bbq_val_clean.jsonl`(1,320)·`bbq_v4_train.json`(57,094) |
| 2. **v4 SFT** | `python train/unsloth/sft_unsloth.py --model Qwen/Qwen3-VL-8B-Instruct --train data/bbq_v4_train.json --out outputs/merged_v4 --epochs 3 --rank 32` | unsloth_env | **~12–14시간** (10,705 step; 시작 시 tqdm ETA 11.8h, 관측 step률 기준 12–14h) | `outputs/merged_v4` (16bit 병합 저장 ~3–5분 포함) |
| 3. 평가 스위트 | `bash infra/run_eval_suite.sh v4 outputs/merged_v4` | lf311 | **~15–20분** (baseline 1,320 + stress 300×6순열 + counterfactual 300) | `data/v4_preds.csv`·`v4_stress.json`·`v4_cf.json` |
| 4. 속도 실측 | 200샘플 `max_new_tokens=4` 측정(0.5초 게이트) | lf311 | **~2–3분** | 초/샘플·8500환산 분 |
| 5. 제출 CSV | `python inference/make_submission.py --model outputs/merged_v4 --test data_build/test_full.csv --images_dir /nonexistent --fallback_image data/placeholder.jpg --out submissions/submission_v4.csv --max_new_tokens 4` | lf311 | **~40–60분** (8,500행 단건 루프, 생성 4토큰) | `submissions/submission_v4.csv` (8,500행) |

- **v4 경로 총합(환경 제외)**: ≈ 데이터 5분 + SFT ~13시간 + 평가 20분 + 속도 3분 + 제출 50분 ≈ **~14시간**.

**v5(GRPO) 단계 — 계획·추정(미실측, v4 merged 완료 후 실행):**

| 단계 | 명령(요약) | venv | 추정시간 | 산출물 |
|---|---|---|---|---|
| 6. **v5 GRPO** | `python train/unsloth/grpo_unsloth.py --base outputs/merged_v4 --out outputs/merged_v5 --val_ids data/val_ids.json --n 2000 --steps 300 --rank 32 --num_gen 4` | unsloth_env | **~2–4시간 (추정)** | `outputs/merged_v5` |
| 7. v5 평가 | `bash infra/run_eval_suite.sh v5 outputs/merged_v5` | lf311 | ~15–20분 | `v5_preds/stress/cf` |
| 8. v5 제출 | `make_submission.py --model outputs/merged_v5 … --max_new_tokens 4` | lf311 | ~40–60분 | `submission_v5.csv` |
| 9. 최종 선택 | `python inference/final_model_selection_report.py --tags v4,v5` | lf311 | ~1분 | v4 vs v5 선택 리포트 |

- **v5 설계(코드 기준)**: base=merged_v4 위 GRPO(vLLM `fast_inference`). 데이터 = BBQ에서 `val_ids` 배제 + v4와 **동일 옵션셔플**(make_bbq_clean 함수 import), amb/dis 균형 2,000 프롬프트. **보상 = accuracy(정답 +2.0 / 오답 −0.5) + format(단일토큰 0/1/2 +0.5 / 위반 −1.0)** → Balanced Accuracy를 직접 최적화. GRPOConfig: lr 5e-6, num_generations=4, grad_accum 4, `max_completion_length=4`, `max_steps=300`, temperature 0.9.
- 시간 근거: 완성 길이 4토큰(매우 짧음)+vLLM 가속 rollout이라 step당 비용 작음, 다만 8B 정책 학습 300 step → **~2–4h 추정**(실측 후 교체).
- **전체 파이프라인 총합(환경 제외)**: v4 ~14h + v5 ~3–5h ≈ **~17–19시간**.
- 주의: `make_submission.py`는 기본 `max_new_tokens=96`이나 **제출은 반드시 `--max_new_tokens 4`** 로 호출(0.5초 컴플라이언스). 단건 추론 루프(배치 없음), 외부 API 0.
- 추론은 `transformers`(lf311) 경로 = 대회 기준환경(A6000) 재현. 학습만 `unsloth_env`.

## 5. 결과 (Results) 〖CAPTURE: 학습/평가 완료 후〗
### 5.1 정직한 Balanced Accuracy (`inference/baseline_eval.py`, 누출제거 `bbq_val_clean.jsonl`)
| 모델 | BA | Acc_ambiguous | Acc_disambiguated | 비고 |
|---|---|---|---|---|
| base (Qwen3-VL-8B) | — | — | — | 미세조정 전 |
| v4 (SFT 단일토큰) | — | — | — | |
| v5 (GRPO) | — | — | — | |
- 카테고리별 BA 분해(성별·인종·나이·종교·외모·장애 등) → Hidden 일반화 대리지표.

### 5.2 추론 속도 (A6000, 200샘플) 〖CAPTURE〗
- v3(추론형 `<think>`) 실측: **1.65초/샘플**(규칙 위반, 기록).
- v4/v5(단일토큰, `max_new_tokens=4`): ___ 초/샘플 → 8,500환산 ___분 [PASS/FAIL 게이트 ≤70분].

### 5.3 강건성 (일반화) 〖CAPTURE〗
- **위치 일관성**(`inference/stress_eval.py`): 6가지 옵션 순열에 대한 답 일관성 ___%.
- **Counterfactual**(`inference/counterfactual_stress.py`): 사회속성 swap 시 답 반전율 ___% (낮을수록 공정).

## 6. 실패 분석 / 절제 연구 (Failure Analysis) — 실제 이력
| 버전 | 시도 | 발견된 문제(근거) | 처방 |
|---|---|---|---|
| v1 | Qwen2.5-VL-7B SFT(LoRA), 가짜 `<think>` | 추론문이 템플릿(실제 추론 아님) | 진짜 CoT 데이터로 교체 |
| **v2** | 진짜 CoT 증류(GPT-4o-mini 5,000건) → **Public 1.0, rank 10** | 자체 BA가 **val 누출로 INFLATED** — 검증셋/추론데이터 생성기의 분할축·RNG 소비순서 불일치, val 배제 코드 부재, `sample_id` 부재로 중복검출 불가 | sha1 SSOT 누출제거 체계 도입 |
| v3 | Qwen2.5-VL 4에폭 | `<think>`로 **1.65초/샘플 = 0.5초 규칙의 3.3배 위반**(실측) | 단일토큰 직답 전환 |
| 검증회의 | 멀티에이전트 7역할 코드 감사 | 위 누출 입증 + 속도 미실측 적발 | "적게, 깨끗하게" 합의(검증된 레버 집중) |
| **v4** | Qwen3-VL-8B + 클린데이터(§3.1) + 단일토큰 | 〖평가 중〗 | 정직한 BA 측정 |
| **v5** | v4 위 GRPO(BA 직접 보상) | 〖평가 중〗 | 정직한 val에서 v4와 비교·선택 |
> 교훈: *만점(v2)을 받고도 그 숫자를 의심한 것이 v4·v5를 만들었다.*

## 7. 재현성 (Reproducibility) — 산출물 ① 충족
- **코드 분리**: 학습 `train/`(unsloth_env) ↔ 추론 `inference/`(lf311). 데이터생성 `data_build/`(API 허용, 추론과 분리).
- **시드 고정**: 데이터 SEED=42, 학습 seed=42, random_state=3407(LoRA init). greedy 디코딩(do_sample=False).
- **외부데이터**: `walledai/BBQ`만(첨부). 표면형 복제 0.
- **재현 절차**: ① `make_bbq_clean.py`(클린데이터+SSOT) → ② `sft_unsloth.py`(v4) → ③ `baseline_eval.py`(정직 BA) → ④ `make_submission.py`(0/1/2, max_new_tokens=4) → ⑤ `grpo_unsloth.py`(v5) → ⑥ `final_model_selection_report.py`(선택).
- **환경 명세**: §4.1 표 전체(OS·CUDA·드라이버·Python·패키지 버전). 인코딩 UTF-8.

## 8. 규칙 준수 체크리스트 (Compliance)
- [x] 베이스 6/1 이전(2025-09)·Apache-2.0 / [x] 오프라인 추론(API 0) / [x] 최종답 LLM 생성(0/1/2) / [x] 학습·추론 코드 분리 / [x] No Leakage(공개 BBQ만·평가셋 미모방·val 누출 제거) / [x] UTF-8·seed·버전 명시 / 〖CAPTURE〗 0.5초/샘플 A6000 실측 PASS.

## 9. 참고문헌 (References)
1. Parrish, A. et al. *BBQ: A Hand-Built Bias Benchmark for Question Answering.* Findings of ACL 2022. arXiv:2110.08193.
2. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. arXiv:2501.12948.
3. Zheng, C. et al. *Large Language Models Are Not Robust Multiple Choice Selectors.* 2023. arXiv:2309.03882.
4. Pezeshkpour, P. & Hruschka, E. *Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions.* 2023. arXiv:2308.11483.
5. Zhang, H. et al. *R-Tuning: Instructing Large Language Models to Say 'I Don't Know'.* 2023. arXiv:2311.09677.
6. Shao, Z. et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. arXiv:2402.03300. (GRPO)

## 부록. 삽입할 캡처 목록 (학습/평가 후)
- 〖CAPTURE-1〗 W&B train/loss·lr·grad_norm 패널 스크린샷.
- 〖CAPTURE-2〗 `baseline_eval.py` 콘솔(정직 BA·카테고리 분해) 출력.
- 〖CAPTURE-3〗 속도 실측 출력(초/샘플·8500환산 분, PASS).
- 〖CAPTURE-4〗 `stress_eval.py`/`counterfactual_stress.py` 강건성 수치.
- 〖CAPTURE-5〗 `final_model_selection_report.py` 최종 선택표.
- ✅ make_bbq_clean 데이터 통계 = §3.1에 실측 기재 완료(고유 58,414 / TRAIN 57,094 / VAL 1,320 / 위치 0·1·2 균등 / 누출 0).
