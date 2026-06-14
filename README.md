# 2026 성균관대 멀티모달 AI Bias 챌린지 — 작업 리포지토리

이미지+텍스트 질의응답(VQA)에서 **근거가 충분하면 정답을 고르고, 근거가 부족하면 "알 수 없음"으로 기권**하는
공정한 멀티모달 AI 모델을 개발한다. 채점지표는 **Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2**.

- 대회: https://dacon.io/competitions/official/236722
- 참가: 솔로 / 클라우드 GPU 대여
- 전략 문서(원본): `~/.claude/plans/https-dacon-io-competitions-official-236-giggly-hummingbird.md`

---

## 폴더 구조

| 폴더 | 용도 | API 허용? |
|---|---|---|
| `data/` | 대회 데이터(예시 샘플, sample_submission, 이미지) + 외부/합성 학습 데이터 | - |
| `data_build/` | 학습 데이터 생성·검증 스크립트 | ✅ GPT API 사용 가능(추론과 분리) |
| `train/` | VLM QLoRA SFT 학습 코드 | - |
| `inference/` | vLLM **오프라인** 추론 코드 (제출 채점용) | ❌ API/인터넷 호출 0 |
| `submissions/` | 제출 CSV (UTF-8) | - |
| `notebooks/` | 폴더 템플릿에서 가져온 참고 노트북 사본 | - |
| `docs/` | 2차 발표자료(PDF) 작업 | - |

> ⚠️ **학습 코드(`train/`)와 추론 코드(`inference/`)는 반드시 분리.** (대회 제출 규칙)
> ⚠️ **추론 단계에서는 외부 API/인터넷 호출이 0이어야 함.** 데이터 생성(`data_build/`)에서만 API 허용.

---

## 대회 규칙 컴플라이언스 체크리스트

- [ ] 베이스 모델 가중치가 **2026-06-01 이전** 공개 (HF 모델카드에서 출시일 확인)
- [ ] 모델 라이선스가 대회 사용 가능 범위 (상업/연구 허용)
- [ ] **추론 시 외부 API/인터넷 호출 0** (오프라인 가중치 직접 로드)
- [ ] **최종 답변은 LLM 생성물** (단순 룰/조건문/정답목록 매핑 금지)
- [ ] 학습 코드 / 추론 코드 **파일 분리**
- [ ] 모든 코드·CSV **UTF-8** 인코딩
- [ ] 외부/합성 데이터가 **평가셋을 모방하지 않음** (Data Leakage 금지)
- [ ] 추론 속도 **샘플당 평균 0.5초 이내** (Test 8,500≈70분 / Hidden 1,500≈13분) — **A6000 기준으로 측정**
- [ ] 기준 환경 재현 가능: RTX A6000 48GB · Python 3.10 · CUDA 12.4 · PyTorch 2.6.0 · Ubuntu 20.04

### 🖥️ GPU 환경 주의 — 학습 vs 평가 분리
- **학습(파인튜닝)**: A100 등 어떤 GPU든 자유. A100 80GB면 배치/모델 여유 ↑.
- **추론(제출 코드)**: 운영진이 **A6000 48GB**에서 재실행 → Private 복원·Hidden 평가.
  - 추론 설정은 반드시 **48GB 이내**에 들어와야 함 (A100 80GB에서만 되는 설정 금지).
  - 7~8B bf16 = 48GB 안전. 32B는 bf16 ~64GB라 초과 → AWQ/GPTQ 4bit 필요(리스크↑).
  - 0.5초/샘플은 **A6000 기준** → A100에서 잴 땐 여유를 두고 판단.
- [ ] Private Score 재현 가능 (seed 고정, requirements/버전 명시)
- [ ] 재학/휴학 증명서 준비 (졸업유예생 불가)

---

## 주요 일정

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
| 2026-06-14 | 프로젝트 폴더 스캐폴딩 | 구조 생성 완료 |
| 2026-06-14 | 대회 포맷 확정(데이터 탭) | 3지선다·인덱스 0/1/2·영어·test 8,500개. 코드 반영 완료 |
| 2026-06-14 | Public 리더보드 확인 | 상위 9팀 만점 1.0, 15위 컷 ~0.998 → Public 포화, Private+Hidden+발표가 승부 |
| 2026-06-14 | 학습 스택 결정 | **LLaMA-Factory SFT(추론형 `<think>`) + TRL GRPO**(VLM). 최신 자료(Downloads) 반영. 문서로 VLM 지원 검증 완료 |
| 2026-06-14 | 키 관리 | W&B/HF/OpenAI 키는 코드 미저장, pod env(.env)로만. (채팅 노출 키는 재발급 권고) |
| | 대회 데이터(open.zip) 다운로드 → data/ | (대기, 본인) |
| | 기권 학습데이터 구축 (data_build) | (대기) |
| | RunPod 학습 → 베이스라인 제출 | (대기, infra/RUNBOOK.md) |

---

## 빠른 시작 (클라우드 인스턴스에서)

```bash
# 1) 환경
pip install -r requirements.txt

# 2) 대회 데이터를 data/ 에 배치 (data/README.md 참조)

# 3) (선택) 학습 데이터 생성 — API 키 필요, 추론과 무관
python data_build/build_abstention_dataset.py --out data/train_abstention.jsonl

# 4) VLM QLoRA 학습
python train/train_vlm_qlora.py --config train/config.yaml

# 5) LoRA 병합
python train/merge_lora.py --adapter outputs/checkpoint-best --out outputs/merged

# 6) 오프라인 추론 → 제출 CSV
python inference/infer_vllm.py --model outputs/merged --test data/test.csv --out submissions/sub_v1.csv
```
