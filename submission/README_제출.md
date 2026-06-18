# 2026 성균관대학교 멀티모달 AI Bias 챌린지 — 2차 제출 산출물
**팀: pk42ac (박성호)** · 최종 모델: **HardNeg-GRPO (v6)** · base: Qwen3-VL-8B-Instruct (Apache-2.0)

---

## 0. 제출물 구성 (요건 매핑)

| 폴더/파일 | 내용 | 대회 요건 |
|---|---|---|
| `code/train/` | 학습 코드 (.py) | 1-① 학습/추론 분리 |
| `code/inference/` | 추론 코드 (.py) — 학습과 분리 | 1-① |
| `code/data_build/` | 외부데이터(공개셋) 생성·정제 스크립트 | 1-② 외부데이터 |
| `code/prompts.py` | 학습/추론 공용 프롬프트 | 1-① |
| `code/requirements.txt` | 의존성(버전 명시) | 1 제출양식 ③ |
| `발표자료_pk42ac.pdf` | 솔루션 발표자료 (15분, PDF) | 2 발표자료 |
| `발표대본_pk42ac.md` | 발표 대본 + Q&A (참고용) | — |
| `V6_SUBMISSION_BUNDLE.md` | 논문(방법·결과) + 재현 코드 단일 번들 | 참고 |
| **[직접 추가 필요] 재학증명서** | 박성호 재학증명서 | **3 참가자격 증빙** |

---

## 1. 개발 / 평가 환경

- **기준 평가환경:** RTX A6000 48GB / **Python 3.10 / CUDA 12.4 / PyTorch 2.6.0 / Ubuntu 20.04**
- 개발은 Python 3.11에서 진행했으나, 추론 코드는 기준 Python 3.10 환경에서 실행 가능.
- **코드·주석 인코딩: UTF-8**
- **외부 API 추론: 없음** — 모델 가중치를 직접 로드해 오프라인으로 실행(대회 규칙 준수).
- 모든 코드는 라이브러리 로딩 포함 오류 없이 실행되도록 구성.

---

## 2. 최종 모델

- **HardNeg-GRPO (v6)** = Qwen3-VL-8B-Instruct(base) + 단일토큰 SFT + 하드네거티브 GRPO
- **가중치(HuggingFace):** https://huggingface.co/psh3333/dacon-skku-bias-vlm-v6
- **추론 결정성:** greedy 디코딩(`do_sample=False`) → 동일 모델·동일 입력이면 출력이 비트단위 동일
  → **Private Score 결정론적 복원** 보장.

---

## 3. Private Score 복원 — 실행 순서

```bash
# 0) 의존성 설치
pip install -r code/requirements.txt

# 1) 누출 제거 클린셋 + val_ids(SSOT) 생성
python code/data_build/make_bbq_clean.py

# 2) v4 단일토큰 SFT → outputs/merged_v4
python code/train/unsloth/sft_unsloth.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --train data/bbq_v4_train.json --out outputs/merged_v4 --epochs 3 --rank 32

# 3) 하드네거티브 풀 채굴 → hardpool.json
python code/data_build/mine_hard.py \
    --base outputs/merged_v4 --out hardpool.json --per 500

# 4) v6 GRPO(최종 모델) → outputs/merged_v6
python code/train/unsloth/grpo_hard_v6.py \
    --base outputs/merged_v4 --out outputs/merged_v6 \
    --hardpool hardpool.json --steps 400 --rank 32 --num_gen 4

# 5) 추론·제출 CSV 생성 (Private Score)
python code/inference/make_submission.py \
    --model outputs/merged_v6 \
    --test_csv data/test.csv --image_root data/test/images \
    --sample_submission data/sample_submission.csv \
    --out submission_v6_final.csv
```

> **가중치를 바로 받아 복원하려면:** HuggingFace `psh3333/dacon-skku-bias-vlm-v6`에서
> 병합 모델을 로드한 뒤 **5)단계만** 실행하면 됩니다(1~4 재학습 불필요).

---

## 4. 핵심 코드 위치 (최종 v6 복원 경로)

- **학습:** `code/train/unsloth/sft_unsloth.py` (SFT) → `code/train/unsloth/grpo_hard_v6.py` (GRPO)
- **추론:** `code/inference/make_submission.py`
- **데이터:** `code/data_build/make_bbq_clean.py`, `code/data_build/mine_hard.py`
- 그 외 `grpo_unsloth.py`(v5)·`sft_reason_v7.py`(v7)·`grpo_robust_v8*.py`(v8a/v8b) 등은
  **ablation·음성결과 재현용**이며, 최종 Private Score 복원에는 위 경로만 필요합니다.

---

## 5. 외부 데이터 (요건 1-②)

모두 **독립 공개 QA 데이터셋**이며, DACON 평가셋에서 파생하지 않았습니다.

| 데이터셋 | 용도 | 출처 |
|---|---|---|
| BBQ (`walledai/BBQ`) | 편향/기권 문항 | HuggingFace |
| SIQA, CommonsenseQA, OpenBookQA, ARC | 일반추론 하드네거티브 | HuggingFace |

- 생성 스크립트: `code/data_build/*.py` (seed 42로 **결정론적 재생성**)
- 누출 방지: sha1 `sample_id`로 val/eval 전역 배제 + assert (`make_bbq_clean.py`)
- ※ 생성된 실데이터 파일(`hardpool.json`, `bbq_v4_train.json`, `val_ids.json`)이 별도로 필요하면
  위 스크립트로 재생성되며, 요구 시 동봉본을 추가합니다.

---

## 6. 제출 체크리스트

- [x] 학습 코드 / 추론 코드 **분리** (`code/train`, `code/inference`)
- [x] `requirements.txt`(라이브러리 버전) + 개발/평가 환경 기재
- [x] 코드·주석 **UTF-8**
- [x] 외부 API 추론 없음 (오프라인 가중치 로드)
- [x] **솔루션 발표자료 PDF** (`발표자료_pk42ac.pdf`)
- [ ] **재학증명서 (박성호)** ← 직접 발급·추가 필요  *(요건 3)*
- [ ] (선택) 생성된 외부데이터 실파일 동봉

---
*최종 모델 = Qwen3-VL-8B-Instruct + SFT + 하드네거티브 GRPO · 외부 API 추론 0 · greedy 결정론 복원*
