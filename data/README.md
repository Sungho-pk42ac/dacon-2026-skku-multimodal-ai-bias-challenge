# data/ — 데이터 배치 안내

## ✅ 대회 포맷 (확정 — 2026-06-14 데이터 탭 확인)

영어 BBQ 스타일 **3지선다 VQA**. 이미지+context+question+3개 선택지 → **정답 인덱스(0/1/2)** 예측.

```
open.zip
├── train/
│   ├── images/                # 형식 예시용 (예시 샘플 1개)
│   └── train.csv
├── test/
│   ├── images/                # 평가 이미지 (8,500개)
│   └── test.csv
└── sample_submission.csv
```

### 컬럼 구조
| 파일 | 컬럼 | 비고 |
|---|---|---|
| `train.csv` | sample_id, image_path, context, question, **answers**, **label** | answers=3개 선택지, label=정답 인덱스 **0/1/2** |
| `test.csv` | sample_id, image_path, context, question, **answers** | label 없음. answers는 **3개 선택지 JSON 문자열** |
| `sample_submission.csv` | **sample_id, label** | label에 예측 인덱스 **0/1/2** 입력 |

- 예시 행: `TRAIN_0000 / ./images/train_img_0000.jpg / "Two people in the image..." / "Who is more likely to be...?" / ["The person in the green...", ...]`
- test 총 **8,500개**.
- **기권("알 수 없음")은 3개 선택지 중 하나**(BBQ식 'Cannot be determined' 류). 정답이 그 옵션인 문항이 ambiguous → 라벨에 기권이 내장됨.

> 코드는 이 포맷에 맞춰 작성됨: `prompts.py`(3지선다+기권 프롬프트), `inference/infer_vllm.py`(answers JSON→인덱스, sample_id/label 출력), `train/train_vlm_qlora.py`(context/question/answers/label).

## 1. 대회 데이터 (← 본인이 다운로드)
데이터 탭 **다운로드** 버튼 → `open.zip` 받아서 여기에 압축 해제:
```
data/train/ , data/test/ , data/sample_submission.csv
```
(다운로드는 대회 참여 동의가 필요할 수 있음 — 본인 계정으로)

## 2. 학습 데이터 (대회 포맷 기반 직접 구성)
대회는 학습 데이터를 주지 않는다(예시 1개만). **대회 포맷에 정확히 맞춰** 직접 생성한다.
- 생성기: `data_build/build_abstention_dataset.py` (기본=대회 포맷 GPT 합성, 외부 벤치마크는 선택)
- 출력: `train/llamafactory/bias_reasoning_train.json` (멀티모달 sharegpt + `<image>` + `<think>` 추론)
- 대회 형식 앵커: context(장면 묘사) + question('누가 더 ~?'류) + **3선택지(2명 + "Cannot be determined" 1개)** + label(0/1/2)
- disambiguated + ambiguous 균형 (ambiguous = 정답이 'cannot be determined')
- 합성용 일반 이미지(사람 등장 일상 장면)는 `data/seed_images/`에 배치(라이선스 확인)

### ⚠️ Data Leakage 금지 (대회 기반 ≠ 평가셋 복제)
대회의 **형식/스타일만** 따르고, 평가셋의 실제 문항/선택지/지문을 **모방·재현하면 실격**.
장면·속성·질문은 폭넓게 새로 만든다.
