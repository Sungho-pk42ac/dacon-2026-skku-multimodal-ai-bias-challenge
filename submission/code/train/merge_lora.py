"""
LoRA 어댑터 + 베이스 모델 병합 (제출용 단일 가중치 생성).

폴더 템플릿 `2. merge and upload.ipynb`의 merge_and_unload 패턴.
* 변경 이유: vLLM 오프라인 추론 시 단일 merged 가중치를 쓰는 게 로딩/속도에 유리.
  (단, multi-LoRA 앙상블을 쓸 경우엔 병합하지 말고 어댑터를 따로 둔다.)
"""

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter", required=True, help="학습된 LoRA 체크포인트 경로")
    parser.add_argument("--out", default="outputs/merged")
    args = parser.parse_args()

    logger.info("베이스 로드: %s", args.base)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    logger.info("어댑터 결합: %s", args.adapter)
    peft_model = PeftModel.from_pretrained(model, args.adapter)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")
    logger.info("병합 완료 → %s", args.out)


if __name__ == "__main__":
    main()
