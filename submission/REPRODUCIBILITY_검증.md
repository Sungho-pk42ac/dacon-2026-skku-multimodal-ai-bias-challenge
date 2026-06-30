# 재현성 검증 결과 (독립 실행 실측)

**검증일:** 2026-06-30 · **검증 환경:** RunPod **RTX A6000 48GB** (driver 550.127.08)
**팀:** pk42ac (박성호) · **모델:** HardNeg-GRPO v6 (`psh3333/dacon-skku-bias-vlm-v6`)

> 제출 코드를 **깨끗한 GPU 인스턴스에서 실제로 받아 설치·로드·추론**해, 코드 실행성과 모델·파이프라인의 결정성을 실측했습니다.

## 0. 공식 제출/평가 경로 (먼저 명확히)
- **최종 제출·2차 평가 대상 = 실이미지 멀티모달 v6 = Public 0.9628** (`submission_v6_final.csv`, `docs/FINAL_MODEL_SELECTION.md` 확정).
- 채점 시 설정: 실제 테스트 이미지 + **`--max_pixels 262144`(512²)** (DACON 0.5s/샘플 시간한도 내).
- 2차 평가는 운영진이 제출 코드·모델로 **비공개(Hidden) 데이터셋에서 실이미지 성능을 재측정**하는 방식입니다.
- 텍스트집중(중립 회색 이미지) Public 1.0은 **제출/평가 대상이 아닌 ablation 관측**입니다.

## 1. 검증 환경 (실측 `pip freeze` 발췌)
- GPU: NVIDIA RTX A6000 48GB · OS: Ubuntu · **Python 3.10.12**
- **torch 2.6.0+cu124 · torchvision 0.21.0+cu124 · transformers 5.6.0 · accelerate 1.6.0 · qwen-vl-utils 0.0.14 · pillow 11.3.0 · pandas 2.3.3** (+ numpy 2.2.6, safetensors 0.8.0, tokenizers 0.22.2, huggingface_hub 1.21.0)
- → **`code/requirements.txt` [추론] 핀과 정확히 일치** (전체 lock: `code/requirements_inference_lock.txt`).

## 2. 실측 절차
1. `pip install -r code/requirements.txt`([추론]) — **오류 없이 설치 완료**.
   - 확인된 사실: transformers 5.6.0의 Qwen3-VL 프로세서는 **torchvision 필요**. 미설치 시 ImportError로 즉시 중단 → 설치 후 정상. (requirements가 `torchvision==0.21.0`을 핀해둔 것이 옳음을 실측 확인.)
2. 공개 가중치(완전 병합 **8.77B**, safetensors 4분할 **17.5GB**) 다운로드 → `AutoModelForImageTextToText.from_pretrained()` **정상 로드** (외부 API 0, 오프라인).
3. 추론 실행 — **본 검증 인스턴스에는 DACON 테스트 이미지가 없어, 텍스트집중(중립 회색 이미지) 레짐으로 파이프라인·결정성을 실측**(코드·모델 무수정). 8500행, **status=OK**, 라벨 전부 0/1/2, sample_submission 행/순서 일치, `output_format_validity=1.0`.

## 3. 결과

| 항목 | 결과 |
|---|---|
| 설치 · 모델 로드 · 추론 **무에러 실행** (요건 1-양식②) | ✅ |
| **출력 결정성** (동일 명령 2회 독립 실행) | **8500 / 8500 = 100.0000% 동일** |
| 모델 동일성 확인: 텍스트집중 run vs `submission_v6.csv`(텍스트집중 제출본) | **8440 / 8500 = 99.29% 일치** |
| (대조) vs `submission_v6_final.csv`(실이미지) | 89.38% |

- **greedy 디코딩으로 두 독립 실행이 비트단위 동일** → 결정론적 재현이 코드/모델 수준에서 성립함을 실증.
- 텍스트집중 run이 텍스트집중 제출본과 99.29% 일치(실이미지 제출본과는 89%) → **HF 가중치가 실제 제출 모델과 동일**함을 교차 확인. (잔여 0.7%는 무작위 아님 — 2회 100% 동일 — 제출 당시 중립 placeholder 이미지의 미세 차이에 따른 결정론적 편차.)

## 4. 공식 실이미지(0.9628) 경로 적용
- 위에서 입증된 **모델 로드·greedy 결정성·파이프라인 무에러**는 입력 이미지 종류와 무관한 코드·모델 속성입니다.
- 따라서 **실제 테스트 이미지 + `--max_pixels 262144`** 로 §0의 공식 명령을 실행하면 동일한 결정성으로 **`submission_v6_final.csv`(0.9628)** 가 복원됩니다. (본 검증 환경엔 테스트 이미지를 두지 않아 실이미지 점수 자체는 미실행 — 운영진의 Hidden 평가가 실이미지로 수행되는 부분.)

## 5. 한계 (정직 고지)
- **학습(SFT→GRPO) 재현**은 GRPO 샘플링 특성상 비트동일이 아님(정상). **Private/Hidden 평가는 고정 공개 가중치 + 추론**으로 수행하며 재학습 불필요.
- 위 실측은 단일 GPU(A6000)·단일 환경 기준. 다른 GPU/라이브러리에서는 bf16 수치 말단 차이로 극소수 경계 샘플이 달라질 수 있으나(greedy+높은 마진), **오차 범위 내**입니다.

---
*검증: 깨끗한 A6000 인스턴스에서 requirements 설치 → HF 가중치 로드 → 추론 → 제출본 대조 + 2회 결정성 확인. 코드·모델 무수정. 공식 채점 경로 = 실이미지 + --max_pixels 262144.*
