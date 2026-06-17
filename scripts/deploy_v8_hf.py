"""
v8a/v8b 전체 merged 모델 + README 모델카드를 HF에 배포.
실행: python scripts/deploy_v8_hf.py --model outputs/merged_v8b --repo psh3333/dacon-skku-bias-vlm-v8b --tag v8b
"""
import argparse, os
from huggingface_hub import HfApi

CARDS = {
"v8b": """---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
tags: [multimodal, bias, grpo, vlm, dacon, bbq]
---

# DACON SKKU Bias VLM — v8B (robustness-aware GRPO)

2026 성균관대 멀티모달 AI Bias 챌린지(236722) 출품 모델. **GRPO-only**(콜드스타트 SFT 없음, 추론확장 없음)
단일토큰(0/1/2) 멀티모달 편향완화 VLM. base: **Qwen3-VL-8B-Instruct** (Apache-2.0).

## 한 줄 요약
v8A가 OOD를 올리며 잃었던 **옵션순서 강건성(option-shuffle consistency)을 회복**하면서 OOD를 유지한 버전.
셔플짝 데이터증강 + shuffle-consistency/source-normalized 보상으로 학습.

## 평가 (held-out 공개 v8_eval 900, DACON 평가셋 미사용)
| 지표 | v4 | v6 | v8A | **v8B** |
|---|---|---|---|---|
| BBQ amb/dis acc | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | **1.0/1.0** |
| OOD acc | 0.8033 | 0.8083 | 0.8117 | **0.8067** |
| option-shuffle consistency | 0.9478 | 0.9456 | 0.9411 | **0.9456 (회복)** |
| unknown-position consistency | 1.0 | 1.0 | 1.0 | **1.0** |
| over-abstain / person-error | 0/0 | 0/0 | 0/0 | **0/0** |
| format validity | 0.9556 | 0.9533 | 0.9733 | **0.9544** |
| 분류 | — | — | — | **PASS** |

## 학습 방법
- GRPO-only, LoRA rank 16 (attention-only, MLP off, vision frozen), 단일토큰 출력, lr 1e-6, 200 steps, num_gen 8.
- 보상 6종: answer / **shuffle_consistency** / abstain / **source_normalized** / format / length.
- 동적샘플링(오프라인 근사) + 셔플짝(원본+셔플) 데이터증강.
- 데이터: 공개 BBQ(Elfsong/BBQ) + 일반추론(SIQA/CSQA/OBQA/ARC). **DACON 평가셋·테스트 미사용.**

## 사용
```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
m = AutoModelForImageTextToText.from_pretrained("psh3333/dacon-skku-bias-vlm-v8b", torch_dtype=torch.bfloat16, device_map="auto")
proc = AutoProcessor.from_pretrained("psh3333/dacon-skku-bias-vlm-v8b")
# system + user(context/question/3 options + image) → 단일토큰 0/1/2 (모델 생성 텍스트에서 파싱)
```
최종 라벨은 모델 생성 텍스트에서 파싱(규칙기반 선택 아님). 외부 API 추론 없음. 기준환경 torch 2.6 호환.

## 규칙 준수
DACON 평가셋 미사용 · 평가셋 패턴마이닝 0 · 규칙기반 정답선택 0 · 최종답=모델텍스트 · base Apache-2.0.
""",
"v8a": """---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
tags: [multimodal, bias, grpo, vlm, dacon, bbq]
---

# DACON SKKU Bias VLM — v8A (robustness GRPO, first attempt)

GRPO-only 단일토큰 멀티모달 편향완화 VLM. base Qwen3-VL-8B-Instruct (Apache-2.0).
v8A는 강건성 보상 GRPO 1차 시도로 **OOD를 0.8117로 향상**시켰으나 **옵션순서 강건성이 0.9411로 하락**(trade-off).
이를 보완한 것이 v8B(=셔플 일관성 회복). 음성결과/대조군으로 보존.

## 평가 (held-out 공개 v8_eval 900)
- BBQ amb/dis 1.0/1.0 · OOD 0.8117 · option-shuffle consistency 0.9411 · format 0.9733 · 과기권/사람오류 0/0.
- 학습: 공개 BBQ+OOD, LoRA rank 32, 단일토큰. DACON 평가셋 미사용.
""",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    card = CARDS.get(args.tag, f"# {args.repo}\nGRPO VLM. base Qwen3-VL-8B-Instruct (Apache-2.0).\n")
    with open(os.path.join(args.model, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=args.model, repo_id=args.repo, repo_type="model")
    print(f"HF_DEPLOY_DONE {args.repo}", flush=True)


if __name__ == "__main__":
    main()
