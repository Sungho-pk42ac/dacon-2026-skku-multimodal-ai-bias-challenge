<div align="center">

# Simpler-is-Better

### GRPO-only로 멀티모달 AI 편향을 완화하다

**2026 성균관대학교 멀티모달 AI Bias 챌린지** · 팀 `pk42ac`
최종 모델 **HardNeg-GRPO** · base **Qwen3-VL-8B-Instruct** (Apache-2.0)

</div>

---

이미지 속 인물을 해석할 때 AI는 **성별·인종 같은 사회적 단서로 편향된 판단**을 내릴 수 있다.
이 프로젝트는 — 근거가 분명하면 답하고, **부족하면 고정관념에 휘둘리지 않고 ‘판단 불가’로 기권**하는 — 공정한 8B 멀티모달 모델을 만든다.

> 평가지표: **Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2**
> 근거가 있을 땐 답하고, 없을 땐 기권하는 *균형 잡힌 정책*만 높은 점수를 받는다.

<br>

## 💡 핵심 주장 — 복잡도는 이득을 더하지 못했다 (동급이면 단순함)

> **콜드스타트 SFT 없이 GRPO만으로** 편향을 제거하면서 일반화를 보존할 수 있다.
> 그리고 — 더 복잡한 RL 변형들은 8B에서 이 단순한 모델을 **유의하게 넘지 못했다.**

콜드스타트(v7)·포화 GRPO(v5)·강건성 보상 5·6종(v8a/v8b)을 전부 시험하고 ablation으로 기각했다.
*동급이면 가장 단순하고 재현성 높은 모델을 택한다* — Occam의 보수적 선택.

| 모델 | 학습 방식 | balanced | OOD | 셔플 일관성 | 판정 |
|---|---|:---:|:---:|:---:|---|
| Sat-GRPO | +GRPO · 포화 분포 | 1.0 | ≈기반 | ≈기반 | 효과 0 (음성) |
| **HardNeg-GRPO ★** | **+GRPO · 하드네거티브** | **1.0** | **0.855** | **0.9544** | **★ 최종 제출** |
| ColdStart | 콜드스타트 + 추론 GRPO | 1.0 | 0.84 | 0.951 | FAIL (음성) |
| Robust-A / B | 강건성 GRPO (r32/r16) | 1.0 | 0.852 | 0.954 / 0.952 | 무이득 (음성) |

<br>

## 🧩 최종 모델은 이렇게 만들어졌다

```
Qwen3-VL-8B-Instruct  →  ① 단일토큰 SFT  →  ② 하드네거티브 GRPO  =  HardNeg-GRPO
     (공개 base)            (누출제거·셔플)        (틀린 문항만 학습)        (최종 제출)
```

- **① SFT** — sha1 sample_id로 검증셋 누출을 전역 차단(SSOT), 옵션 순서를 셔플해 *위치편향*("애매하면 2번")을 깨고 0/1/2 직답을 학습.
- **② GRPO** — base가 *틀리는* 일반추론 문항(SIQA·CSQA·OBQA·ARC)만 모아 학습 → 보상 신호가 살아 정책이 실제로 개선되고, 일반추론 하드네거티브가 OOD 일반화를 끌어올린다. 보상은 규칙기반 2종(정답 ±, 형식 ±)뿐 — 신경망 보상모델 없음.
- **추론** — greedy 디코딩이라 동일 입력이면 출력이 비트단위 동일. 최종 라벨은 *모델 생성 텍스트*를 파싱(룰 선택 아님).
- **가중치** — `psh3333/dacon-skku-bias-vlm-v6`

<br>

## ✅ 재현 검증 (RunPod A6000 실측)

공개 데이터만으로 자기완결 재현을 확인했다 — DACON 정답 라벨 불필요.

| 검증 항목 | 결과 |
|---|---|
| 텍스트집중 balanced accuracy | **1.0 재현** (공개 BBQ held-out) |
| **실이미지 제출 (DACON test)** | 기존 0.9628 제출과 **예측 99.29% 일치** → *오차 범위 내 복원* ✅ |
| 학습 코드 (Unsloth + Qwen3-VL) | SFT 정상 시작·학습 ✅ |
| 추론 속도 | ~0.22s/샘플 (A6000, 0.5s 예산 내) ✅ |

<br>

## 🚀 재현하기

```bash
pip install -r requirements.txt          # 학습 재현 시 unsloth 추가 설치

# 가중치를 받아 추론만 복원 (가장 빠름)
python inference/make_submission.py --model psh3333/dacon-skku-bias-vlm-v6 \
    --test_csv data/test.csv --image_root data/test \
    --sample_submission data/sample_submission.csv \
    --out submission.csv --max_pixels 262144

# 처음부터 학습 (선택)
python data_build/make_bbq_clean.py                                              # 클린셋 + val_ids
python train/unsloth/sft_unsloth.py --train data/bbq_v4_train.json --out outputs/merged_v4 --epochs 3 --rank 32
python data_build/mine_hard.py --base outputs/merged_v4 --out hardpool.json --per 500
python train/unsloth/grpo_hard_v6.py --base outputs/merged_v4 --out outputs/merged_v6 --hardpool hardpool.json --steps 400
```

> 기준 평가환경: RunPod **A6000 48GB · Python 3.10 · CUDA 12.4 · PyTorch 2.6.0**. 추론은 외부 API·인터넷 호출 0.

<br>

## 📂 저장소 구조

| 경로 | 내용 |
|---|---|
| [`experiments/`](experiments/) | **실험(v4~v8b)별 인덱스** — 버전마다 코드·데이터·학습명령·결과·판정 카드 |
| `prompts.py` | 프롬프트·기권 패턴·`text_to_label` 파서 (학습·추론 공통) |
| `data_build/` | 외부데이터(공개 QA) 생성·누출제거 — `make_bbq_clean.py` · `mine_hard.py` |
| `train/unsloth/` | 학습 — `sft_unsloth.py`(SFT) · **`grpo_hard_v6.py`(최종 GRPO)** · v5/v7/v8 변형(ablation) |
| `inference/` | 오프라인 추론·평가 — `make_submission.py` · `eval_suite.py` |
| `submission/` | 제출 패키지: `code/` + `README_제출.md` + 발표자료 PDF + [`V6_SUBMISSION_BUNDLE.md`](submission/V6_SUBMISSION_BUNDLE.md)(논문+재현 코드) |
| `infra/` | RunPod 셋업 · `smoke_test_v6.py` · `eval_open.py` |
| `docs/` | 실험 방법론·결과 리포트 |

<br>

## 📑 더 보기

- **🧪 실험 인덱스 (v4~v8b)** — [`experiments/`](experiments/README.md) · 버전별 코드·결과 한눈에
- **논문 + 재현 코드 번들** — [`submission/V6_SUBMISSION_BUNDLE.md`](submission/V6_SUBMISSION_BUNDLE.md)
- **발표자료 (PDF, 34p)** — [`submission/발표자료_pk42ac.pdf`](submission/발표자료_pk42ac.pdf)
- **전체 비교(v4~v8b)** — [`docs/V4_V8B_COMPLETE_COMPARISON.md`](docs/V4_V8B_COMPLETE_COMPARISON.md)

<div align="center">
<sub>base: Qwen3-VL-8B-Instruct (Apache-2.0) · 외부 API 추론 0 · greedy 결정론 복원 · 팀 pk42ac · 2026</sub>
</div>
