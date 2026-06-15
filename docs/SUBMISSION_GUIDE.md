# 2차 산출물 제출 / 재현 가이드

> 2026 성균관대 멀티모달 AI Bias 챌린지 — 코드 검증 및 Private Score 복원용 문서
> 제출처: dacon@dacon.io / 제목: `[팀명] 2026 성균관대학교 멀티모달 AI 챌린지 최종 산출물 제출` / 기한: ~7.2(목) 10:00

---

## 0. 제출 패키지 구성 (체크리스트)

- [ ] **추론 코드**(inference/) + **학습 코드**(train/, data_build/) — 분리됨 ✅
- [ ] **모델 가중치** — 최종 선택 모델(merged_v4 또는 merged_v5) / HF: `psh3333/dacon-skku-bias-vlm-*`
- [ ] **외부 데이터** — walledai/BBQ (공개, CC-BY) 사용. 합성 데이터 없음(v4 기준)
- [ ] **requirements.txt** + 본 가이드(개발환경·라이브러리 버전)
- [ ] **발표자료 PDF**(15분) — `docs/` 참조
- [ ] **재학/휴학 증명서** (본인 발급 필요)

---

## 1. 모델 카드 (규칙 3 — 2026-06-01 이전 공개 확인)

| 항목 | 값 |
|---|---|
| 베이스 모델 | **Qwen/Qwen3-VL-8B-Instruct** |
| 가중치 공개일 | **2025-09-23** (✅ 2026-06-01 이전) |
| 라이선스 | **Apache-2.0** (상업/연구 허용) |
| 다운로드 경로 | `https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct` |
| 파인튜닝 | LoRA SFT (단일토큰 직답) + (선택) GRPO |

## 2. 기준 평가환경 / 라이브러리

- OS: Ubuntu 20.04 / Python 3.10 / CUDA 12.4 / PyTorch 2.6.0 / RTX A6000 48GB
- 핵심 추론 의존성: `transformers==5.6.0`, `qwen-vl-utils==0.0.14`, `accelerate==1.6.0` (전체 `requirements.txt`)
- ⚠️ Qwen3-VL은 transformers ≥4.57 필요 → 기준환경에 **transformers 5.6.0 설치** 필수

## 3. 코드 맵 (학습/추론 분리 — 규칙 1·5)

```
[학습 train]
  data_build/make_bbq_clean.py     # BBQ→대회포맷, 누출제거(val_ids SSOT), 단일토큰 타깃
  train/llamafactory/qwen3vl_lora_sft.yaml   # LLaMA-Factory SFT 설정(template qwen3_vl_nothink)
  train/grpo/train_grpo_vlm.py     # (선택) GRPO 강화학습
  prompts.py                       # 공용 프롬프트(학습/추론 동일 소스)

[추론 inference] — 외부 API/인터넷 호출 0
  inference/make_submission.py     # test.csv → 제출 CSV (오프라인, model.generate)
  inference/baseline_eval.py       # 자체 검증 BA 측정
  prompts.py                       # (공용)
```
> ⚠️ 사용 안 하는 레거시 파일(`train/train_vlm_qlora.py`, `inference/infer_vllm.py`, `train/config.yaml`, `train/merge_lora.py`)은 최종 제출 전 제거 또는 `legacy/`로 분리 권장.

## 4. Private Score 복원 절차 (재현)

```bash
# 1) 환경
pip install -r requirements.txt
# (학습 재현 시) LLaMA-Factory 소스 설치

# 2) 학습 데이터 생성 (공개 BBQ → 대회 포맷, 누출제거, seed=42 고정)
python data_build/make_bbq_clean.py        # → data/bbq_v4_train.json, val_ids.json, bbq_val_clean.jsonl

# 3) SFT 학습 (LoRA)
llamafactory-cli train train/llamafactory/qwen3vl_lora_sft.yaml   # → outputs/sft_v4
llamafactory-cli export --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_name_or_path outputs/sft_v4 --template qwen3_vl_nothink \
  --finetuning_type lora --export_dir outputs/merged_v4

# 4) (선택) GRPO 강화학습
python train/grpo/train_grpo_vlm.py --base outputs/merged_v4 --out outputs/grpo_v5
#   → 병합: llamafactory-cli export ... --model_name_or_path outputs/merged_v4 --adapter_name_or_path outputs/grpo_v5

# 5) 추론 → 제출 CSV (오프라인)
python inference/make_submission.py --model outputs/merged_v4 \
  --test <대회 test.csv> --images_dir <test 이미지 폴더> --out submissions/submission.csv --max_new_tokens 4
```
- 시드 고정: 모든 스크립트 `SEED=42`. 학습/추론 분리, CSV UTF-8.

## 5. 규칙 준수 요약

| 규칙 | 준수 |
|---|---|
| 모델 6/1 이전 공개·오픈소스 | ✅ Qwen3-VL-8B(2025-09, Apache-2.0) |
| 추론 시 외부 API/인터넷 0 | ✅ 로컬 가중치 `model.generate` |
| 최종답 = LLM 생성 | ✅ 모델이 인덱스 토큰 직접 생성(룰 매핑 아님) |
| 학습/추론 코드 분리 | ✅ train/·data_build/ vs inference/ |
| Data Leakage 금지 | ✅ 공개 BBQ만, 평가셋 형식 모방·합성 안 함, val 누출 제거 |
| 0.5초/샘플 @ A6000 | ⏳ 단일토큰으로 충족 목표 — **A6000 실측치 기입**: ___ s/sample |

## 6. 최종 성능 (결과 확정 후 기입)

| 모델 | 자체검증 BA(bbq_val_clean) | 속도(s/sample, A6000) | Public | 최종선택 |
|---|---|---|---|---|
| v4 (SFT) | ___ | ___ | ___ | ☐ |
| v5 (GRPO) | ___ | ___ | ___ | ☐ |

> 최종 채점 파일 = 정직한 검증 BA 최고 + 0.5초 통과 모델.
