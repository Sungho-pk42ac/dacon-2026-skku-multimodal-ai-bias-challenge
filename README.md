# 2026 성균관대 멀티모달 AI Bias 챌린지 — 작업 리포지토리

이미지+텍스트 질의응답(VQA)에서 **근거가 충분하면 정답을 고르고, 근거가 부족하면 "알 수 없음"으로 기권**하는
공정한 멀티모달 AI 모델을 개발한다. 채점지표는 **Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2**.

- 대회: https://dacon.io/competitions/official/236722
- 참가: 솔로 / 클라우드 GPU 대여(RunPod A6000 48GB = 기준환경 일치)
- 전략 문서: [`STRATEGY.md`](STRATEGY.md) · 일정: [`SCHEDULE.md`](SCHEDULE.md) · 운영: [`infra/RUNBOOK.md`](infra/RUNBOOK.md)

> **현황 (2026-06-15)**: v2 **Public 1.0(만점)**·리더보드 10위(2차 진출권). HF 배포 완료.
> **멀티에이전트 회의로 전략 재정립** → 🔴 *자체검증 BA 0.97은 val 누출로 부풀려짐(INFLATED)* 발견.
> 현재 **확정 노선 = "적게, 깨끗하게"**:
> - **v4** = **Qwen3-VL-8B-Instruct**(2025-09·Apache-2.0) + 누출제거(SSOT) + **단일토큰 직답**(0.5초 컴플라이언스).
> - **v5** = v4 위 **GRPO**(정답·기권 보상 = Balanced Accuracy 직접 최적화). v4 vs v5를 *정직한 val*에서 비교해 최종 선택.
> - 데이터 = **공개 BBQ만**(증강 없음, 누출 0). 검증 = 카테고리별 BA(Hidden 일반화 대리).
> 자동 파이프라인: v3 → v4 → v5 (pod 자율). 회의록·전략: [`docs/FINAL-STRATEGY.md`](docs/FINAL-STRATEGY.md)

---

## 폴더 구조

| 폴더/파일 | 용도 | API 허용? |
|---|---|---|
| `prompts.py` | **공유 모듈**: system/user 프롬프트, 기권 패턴, `text_to_label` 파서 (학습·추론 공통) | - |
| `data/` | 대회 데이터 + 변환된 BBQ 학습/검증 데이터(`bbq_train.json`, `bbq_val.jsonl`, `bbq_reason_train.json`) | - |
| `data_build/` | BBQ→대회포맷 변환·진짜 CoT 생성 스크립트 (`make_bbq_data.py`, `make_bbq_reasoning.py`) | ✅ GPT API 사용 가능(추론과 분리) |
| `train/llamafactory/` | **메인 학습**: LLaMA-Factory QLoRA SFT 설정(`qwen25vl_lora_sft.yaml`, `dataset_info.json`) | - |
| `train/grpo/` | **강화학습**: TRL GRPO VLM 학습(`train_grpo_vlm.py`, 정답 일치 보상) | - |
| `inference/` | **오프라인** 추론·자체검증 (`make_submission.py`, `baseline_eval.py`) | ❌ API/인터넷 호출 0 |
| `infra/` | RunPod SSH 헬퍼·오케스트레이터(`pod_master*.sh`)·환경 셋업·`RUNBOOK.md` | - |
| `submissions/` | 제출 CSV (UTF-8) — `submission_v1.csv`, `submission_v2.csv` | - |
| `notebooks/` | 폴더 템플릿에서 가져온 참고 노트북 사본 | - |
| `docs/` | 2차 발표자료(PDF) 작업 | - |

> ⚠️ **학습 코드(`train/`)와 추론 코드(`inference/`)는 반드시 분리.** (대회 제출 규칙)
> ⚠️ **추론 단계에서는 외부 API/인터넷 호출이 0이어야 함.** 데이터 생성(`data_build/`)에서만 API 허용.

---

## 대회 규칙 컴플라이언스 체크리스트

- [x] 베이스 모델 가중치가 **2026-06-01 이전** 공개 — Qwen2.5-VL-7B-Instruct(2025.01) ✓
- [x] 모델 라이선스가 대회 사용 가능 범위 — Qwen Apache-2.0 ✓
- [x] **추론 시 외부 API/인터넷 호출 0** — transformers 오프라인 가중치 직접 로드 ✓
- [x] **최종 답변은 LLM 생성물** — `model.generate` 출력을 `text_to_label`로 파싱(룰 매핑 아님) ✓
- [x] 학습 코드 / 추론 코드 **파일 분리** — `train/` vs `inference/` ✓
- [x] 모든 코드·CSV **UTF-8** 인코딩 ✓
- [x] 외부/합성 데이터가 **평가셋을 모방하지 않음** — 공개 BBQ 벤치마크 기반 ✓
- [ ] 추론 속도 **샘플당 평균 0.5초 이내** — ⚠️ 현재 ~0.88초(`<think>` 때문) → **제출용 단축 필요**
- [x] 기준 환경 재현 가능 — RunPod A6000 48GB(Py3.10/CUDA12.4/torch2.6/Ubuntu) ✓ (추론 base env 기준)

### 🖥️ GPU 환경 주의 — 학습 vs 평가 분리
- **학습(파인튜닝)**: A100 등 어떤 GPU든 자유. A100 80GB면 배치/모델 여유 ↑.
- **추론(제출 코드)**: 운영진이 **A6000 48GB**에서 재실행 → Private 복원·Hidden 평가.
  - 추론 설정은 반드시 **48GB 이내**에 들어와야 함 (A100 80GB에서만 되는 설정 금지).
  - 7~8B bf16 = 48GB 안전. 32B는 bf16 ~64GB라 초과 → AWQ/GPTQ 4bit 필요(리스크↑).
  - 0.5초/샘플은 **A6000 기준** → A100에서 잴 땐 여유를 두고 판단.
- [ ] Private Score 재현 가능 (seed 고정, requirements/버전 명시)
- [ ] 재학/휴학 증명서 준비 (졸업유예생 불가)

---

## 주요 일정 (전체·체크리스트는 [`SCHEDULE.md`](SCHEDULE.md))

| 날짜 | 이벤트 |
|---|---|
| 06.22 | 팀 병합 마감 (솔로는 무관) |
| **06.29 10:00** | 대회 종료 = 리더보드(1차) 마감 |
| 07.02 10:00 | 2차 평가 대상자(상위 15팀) 산출물 제출 마감 |
| 07.10 | 2차 평가·코드검증 / 시상식 대상자 안내 |
| 07.14 | 오프라인 시상식 (참석 필수) |

---

## 진행 로그

| 날짜 | 작업 | 결과/메모 |
|---|---|---|
| 06-14 | 프로젝트 스캐폴딩 + 대회 포맷 확정 | 3지선다·인덱스 0/1/2·영어·test 8,500개. 코드 반영 완료 |
| 06-14 | Public 리더보드 확인 | 상위 9팀 만점 1.0, 15위 컷 ~0.998 → Public 포화, Private+Hidden+발표가 승부 |
| 06-14 | 학습 스택 결정 | **LLaMA-Factory SFT(추론형 `<think>`) + TRL GRPO**(VLM). base=Qwen2.5-VL-7B-Instruct |
| 06-14 | BBQ → 대회포맷 변환 | walledai/BBQ(58k) → train 57k 균형 + 자체검증 `bbq_val.jsonl`(1,320 균형) |
| 06-14 | **v1** SFT(LoRA) → 병합 → 자체검증 | 자체검증 BA **0.9054(base) → 0.9682** (disambiguated 0.86→0.97) |
| 06-14 | 진짜 CoT 데이터 생성 | gpt-4o-mini로 실제 추론문 5,000건(`bbq_reason_train.json`) — v1의 가짜 `<think>` 대체 |
| 06-15 | **v2** SFT(진짜 추론 데이터) → 제출 | **Public 1.0(만점), 리더보드 10위 → 2차 진출권** ✅ |
| 06-15 | v2 HF 배포 | `psh3333/dacon-skku-bias-vlm-v2` (private) |
| 06-15 | **v3** SFT(Qwen2.5, 4에폭) | 속도 **1.65초/샘플**(규칙 0.5초 위반) → 폐기 후보. Public 참고용만 |
| 06-15 | **멀티에이전트 회의**(`team` 스킬, 7에이전트) | OSS~20+자산+레포 검토 → 🔴 **자체검증 BA 0.97 = val 누출(INFLATED)** 발견. 합의 "적게, 깨끗하게" → [`.omc/research/FINAL-STRATEGY.md`] |
| 06-15 | **베이스 교체 확정** | Qwen2.5-VL-7B → **Qwen3-VL-8B-Instruct**(2025-09, Apache-2.0). lf311 호환 검증(transformers 5.6, `qwen3_vl_nothink`) |
| 06-15 | **v4 확정**(클린) | `make_bbq_clean.py`(sha1 sample_id→`val_ids.json` SSOT, 누출제거) + 단일토큰 직답 + 0.5초 실측 게이트. 자동 큐 |
| 06-15 | **v5 확정**(GRPO) | merged_v4 위 GRPO(trl1.6/grpo3_env), 누출배제, 정직한 val에서 v4와 비교. 자동 큐 |
| 06-15 | 카테고리별 BA 평가 추가 | `baseline_eval.py` 카테고리 분해(Hidden 일반화 대리지표) |
| 06-15 | 데이터 증강 방침 | **공개 BBQ만 유지**(SB-Bench 등 외부 미사용) — Hidden 중복=실격 리스크 회피 |
| 06-15 | 2차 제출자료 | `requirements.txt`·`docs/SUBMISSION_GUIDE.md`·`docs/presentation_outline.md` 골격 작성 |
| 06-15 | 키 관리 | W&B/HF/OpenAI 키는 코드 미저장, pod env로만. (채팅 노출 키는 재발급 권고) |

---

## 실제 파이프라인 (RunPod 기준)

> 환경 2개 분리: 학습=py3.11 venv `/workspace/lf311`(LLaMA-Factory) · GRPO=py3.11 venv `/workspace/grpo_env`(TRL 0.19, llm_blender 충돌 회피) · 추론=base py3.10.
> RunPod는 프록시 SSH라 `infra/pod.sh`(`-tt`+stdin 파이프)로 원격 실행, 코드 전송은 GitHub 배포키로 pod가 git pull. 상세는 [`infra/RUNBOOK.md`](infra/RUNBOOK.md).

```bash
# 1) 학습 데이터 생성 (data_build, API 허용 — 추론과 분리)
python data_build/make_bbq_data.py          # BBQ → 대회포맷(sharegpt) + 자체검증셋
python data_build/make_bbq_reasoning.py      # gpt-4o-mini 진짜 CoT 5,000건

# 2) SFT 학습 (LLaMA-Factory, lf311 env)
llamafactory-cli train train/llamafactory/qwen25vl_lora_sft.yaml

# 3) LoRA 병합 → merged 가중치
llamafactory-cli export ...                  # → outputs/merged_v2

# 4) 자체검증 BA 측정 (오프라인)
python inference/baseline_eval.py --model outputs/merged_v2 --val data/bbq_val.jsonl

# 5) 제출 CSV 생성 (전체 8,500, 오프라인)
python inference/make_submission.py --model outputs/merged_v2 \
  --test data_build/test_full.csv --fallback_image data/placeholder.jpg \
  --out submissions/submission_v2.csv

# 6) (선택) GRPO 강화학습 — grpo_env
python train/grpo/train_grpo_vlm.py --base outputs/merged_v2 --out outputs/grpo_v4
```

**자율 오케스트레이션**: 위 2~5단계를 한 번에 도는 `infra/pod_master.sh <tag> <dataset> <epochs>`,
GRPO 전체 사이클은 `infra/pod_master_grpo.sh` (nohup으로 PC 꺼도 진행, 완료마커 `/workspace/MASTER_*_DONE`).

> ⚠️ 추론 속도가 현재 ~0.88초/샘플로 **0.5초 예산 초과** (`<think>` 추론 토큰 때문). 2차 제출 전 제출용은 답만 출력하도록 단축 필요.
