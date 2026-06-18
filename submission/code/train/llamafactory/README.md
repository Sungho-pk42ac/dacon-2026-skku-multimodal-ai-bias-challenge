# LLaMA-Factory 학습 (메인 경로)

Qwen2.5-VL LoRA SFT (추론형 `<think>`) → 병합 → (선택) GRPO → vLLM 추론.

## 0. 설치 (pod, torch는 대회환경 일치)
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[torch,metrics]" && pip install wandb qwen-vl-utils
# 멀티모달 추가 의존성 필요 시: pip install -e ".[torch,metrics,vllm]"
```

## 1. 데이터 연결
- `data_build/build_abstention_dataset.py` 로 `train/llamafactory/bias_reasoning_train.json` 생성.
- LLaMA-Factory의 `data/dataset_info.json` 에 우리 `dataset_info.json` 항목(`bias_reasoning`)을 합치거나,
  실행 시 `--dataset_dir train/llamafactory` 로 우리 폴더를 가리킨다.

## 2. 템플릿 확인 (중요)
설치된 버전의 Qwen2.5-VL 템플릿명을 먼저 확인:
```python
from llamafactory.data.template import TEMPLATES
print([t for t in sorted(TEMPLATES) if "qwen" in t])   # qwen2_vl / qwen2_5_vl 중 존재하는 것 사용
```
→ `qwen25vl_lora_sft.yaml` 의 `template:` 값을 맞춘다.

## 3. 학습 실행
```bash
set -a; source ../.env; set +a          # WANDB_API_KEY, HF_HOME 등
llamafactory-cli train ../train/llamafactory/qwen25vl_lora_sft.yaml
```
- W&B 대시보드로 loss/eval 실시간 전송 (run_name: bias-sft-qwen25vl)

## 4. 병합 (vLLM 추론용 단일 가중치)
```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_name_or_path outputs/sft_qwen25vl_lora \
  --template qwen2_vl --finetuning_type lora \
  --export_dir outputs/merged --export_size 4
```

## 5. (선택) HF private 업로드
```bash
huggingface-cli upload --private psh3333/dacon-bias-qwen25vl outputs/merged
```
⚠️ 반드시 **private** (대회 산출물 보호). 토큰은 env(HF_TOKEN)/`huggingface-cli login`.

## 6. (Tier-3) GRPO
SFT 병합 모델을 시작점으로:
```bash
python ../train/grpo/train_grpo_vlm.py --sft_model outputs/merged \
  --dataset ../data/grpo_train.json --out outputs/grpo_qwen25vl
```

## 7. 추론 → 제출
```bash
python ../inference/infer_vllm.py --model outputs/merged \
  --test ../data/test.csv --images_dir ../data/test/images --out ../submissions/sub_v1.csv
```
⚠️ 추론형(`<think>`)은 토큰이 늘어 **느려짐** → `infer_vllm.py`의 max_tokens·image 해상도로 0.5초(A6000) 맞추기.
   리더보드 속도 빠듯하면 제출용은 think 생략(답만) 버전도 고려, Novelty용 think 버전은 2차 발표에.
