# DACON 규칙 준수 체크리스트 — v8B

> 2026 성균관대 멀티모달 AI Bias 챌린지(236722). 본 문서는 v8B 파이프라인이 대회 규칙을
> 준수함을 코드 근거와 함께 확인한다. (✅ 충족 / ⏳ 조건부 / 해당 산출물 경로 명시)

| # | 규칙 | 상태 | 근거 |
|---|---|---|---|
| 1 | 외부/원격 API 추론 없음 | ✅ | 모든 추론이 로컬 transformers(`inference/make_submission.py`, `inference/eval_suite.py`, `inference/infer_unified.py`). vLLM은 학습 가속에만, 제출 추론은 transformers(기준환경 torch 2.6). API 호출 코드 없음. |
| 2 | DACON 테스트/비공개/히든 데이터를 학습에 미사용 | ✅ | 학습 풀(`data_build/build_v8b_pool.py`)은 공개데이터(Elfsong/BBQ, SIQA, CSQA, OBQA, ARC)만 사용. test_full.csv는 학습/채굴/보상/프롬프트에 일절 미참조. |
| 3 | 평가셋 패턴 마이닝 없음 | ✅ | 채굴은 공개 후보에 한정. `val_ids.json` + `v8_eval.json` 항목은 풀에서 `sid()`로 제외 → 학습/평가 분리. DACON 평가셋 통계/패턴 추출 없음. |
| 4 | 규칙기반 최종답 선택 없음 | ✅ | 최종 라벨은 `prompts.text_to_label()`로 **모델 생성 텍스트**를 0/1/2로 정규화만. 규칙/휴리스틱으로 정답을 고르지 않음(텍스트 파싱뿐). |
| 5 | 최종 라벨 = 모델 생성 출력 | ✅ | `make_submission.py`: `label = text_to_label(raw_output, answers)`. preview JSONL에 행별 `raw_output` + `parsed_label` 기록(감사 가능). |
| 6 | 학습/추론 코드 분리 | ✅ | 학습 `train/unsloth/grpo_robust_v8b.py`, 추론 `inference/make_submission.py`·`eval_suite.py`. 상호 import 없음. |
| 7 | 외부 데이터셋 문서화 | ✅ | `hf/v8b_package/metadata.json.training_data_sources` + `docs/V8B_POOL_REPORT.md`에 출처·라이선스 명시. |
| 8 | base 모델 공개일/라이선스 문서화 | ✅ | Qwen3-VL-8B-Instruct, Apache-2.0. `metadata.json.base_model_license/base_model_release_date/base_model_url`. (대회 허용 2026-06-01 이전 공개 — 최종 확인 필요 항목으로 표기) |
| 9 | 추론 환경 문서화 | ✅ | 기준환경 RTX A6000 / Py3.10 / CUDA12.4 / PyTorch 2.6.0 / Ubuntu20.04. 제출 추론은 lf311(transformers 5.0/torch 2.6.0+cu124, 기준환경 일치). `metadata.json`에 torch/cuda/python 기록. |
| 10 | 제출 CSV 검증 통과 | ⏳ | `submissions/submission_v8b_validation.json` 생성. **실제 테스트 이미지 확보 후** 생성 시 status=OK 확인(행수·순서·라벨 0/1/2·null 0). 현재 pod에 실제 이미지 부재 → 이미지 배치 후 생성. |
| 11 | HF 패키지에 비공개 평가데이터 미포함 | ✅ | `scripts/export_to_hf.py`는 LoRA 어댑터·문서·메타만 패키징. DACON 테스트/평가 원본·키 미포함. 업로드는 `--push_to_hub` 명시 시에만, 기본 private. |

## 이미지 처리 규칙 준수 (사용자 지정)

| 규칙 | 상태 | 근거 |
|---|---|---|
| DACON 제출 추론은 실제 테스트 이미지 사용(image_path + --image_root) | ✅(코드) | `make_submission.py resolve_image_path()` 실제 이미지 해석. |
| 제출에 placeholder 고정이미지 금지 | ✅ | `--allow_placeholder` 없이는 placeholder fallback 발생 시 **즉시 중단(exit 2)** + 검증 INVALID. |
| placeholder는 공개 텍스트데이터(image_path=null)만 허용 | ✅ | 학습/평가(BBQ·OOD)는 placeholder 사용(규칙 허용). 제출 경로와 분리. |
| DACON 테스트 이미지 학습/채굴/보상/프롬프트 미사용 | ✅ | 학습·채굴 전부 공개데이터 placeholder. test 이미지 미참조. |
| real_image_loaded_count / placeholder_fallback_count 기록 | ✅ | `submission_v8b_validation.json`에 기록. |
| placeholder_fallback_count>0 이면 중단·보고 | ✅ | 위와 동일(exit 2 + INVALID_PLACEHOLDER_USED). |

## 현재 블로커
- **실제 DACON 테스트 이미지(test_img_*.jpg, 8500개)가 pod에 부재** → 제출 CSV는 이미지 배치 후 생성.
  나머지(학습·평가·HF·리포트)는 공개데이터 기반이라 영향 없이 완료.
