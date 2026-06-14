# RunPod 학습 런북 (사용자 직접 수행 단계 포함)

> 결제·로그인·pod 대여는 **본인이** 해야 합니다(계정/비용). 나머지는 스크립트가 자동화.

## 0. 선행조건 (이게 안 되면 학습 의미 없음)
- [ ] 대회 평가 데이터 + 제출 형식 확보 → `data/` (data/README.md Day 0 체크리스트)
- [ ] 기권 학습 데이터 `data/train_abstention.jsonl` 구축 (data_build/)
- [ ] 베이스 모델 출시일 6/1 이전 + 라이선스 확인

## 1. RunPod에서 pod 대여 (← 본인)
1. runpod.io 로그인 (직접)
2. **GPU 선택**:
   - 학습: **H100 80GB**(가장 빠름, 반복 실험 유리) 또는 **A100 80GB**(7~8B QLoRA엔 충분). 예산 절약은 A100 40GB/A6000도 가능.
   - ⚠️ **추론 검증은 A6000 48GB 한도** 안에서 (운영진 재현 환경). H100 80GB에서만 되는 설정 금지.
   - ⚠️ **0.5초/샘플은 A6000 기준.** H100은 A6000보다 2~3배 빠르므로, H100 측정치에 여유를 두고 판단(가능하면 A6000 pod로 최종 속도 검증).
3. **템플릿**: PyTorch + **CUDA 12.4** (대회 기준 환경에 맞춤), 디스크 100GB+ (모델 가중치 큼)
4. 포트: Jupyter/SSH 활성화

## 2. 프로젝트 업로드 (← 본인, 택1)
- (A) RunPod 웹 터미널/Jupyter로 이 폴더(`dacon-bias-challenge`) 통째 업로드 → `/workspace/`
- (B) GitHub에 push 후 pod에서 `git clone` (가중치/데이터는 .gitignore라 별도 업로드)
- 데이터(`data/`)는 용량 커서 보통 별도 업로드 또는 pod에서 직접 생성

## 3. 환경변수 (← 본인, 키는 채팅 아닌 pod에서)
```bash
cd /workspace/dacon-bias-challenge
cp infra/env.example .env && nano .env     # WANDB_API_KEY / HF_TOKEN / OPENAI_API_KEY 입력
set -a; source .env; set +a
```
> ⚠️ 키는 .env(=.gitignore)에만. 절대 커밋 금지. 노출된 키는 재발급.

## 4. 셋업·학습 실행 (메인 = LLaMA-Factory)
- 상세: `train/llamafactory/README.md`
```bash
# (1) 데이터 생성 (C방식, OPENAI_API_KEY 필요)
python data_build/build_abstention_dataset.py --synth_num 2000 --hf_benchmark <공개셋>
# (2) LLaMA-Factory 설치 후 SFT (추론형)
llamafactory-cli train train/llamafactory/qwen25vl_lora_sft.yaml
# (3) 병합 → (선택) GRPO → 추론
```
> 백업 경로(TRL): `bash infra/runpod_setup.sh` (단일 스크립트, train_vlm_qlora.py 사용)

## 5. 결과
- SFT LoRA → `outputs/sft_qwen25vl_lora` / 병합 → `outputs/merged`
- (선택) GRPO → `outputs/grpo_qwen25vl`
- 제출 CSV → `submissions/sub_v1.csv` → DACON `제출` 탭 업로드(본인)
- (선택) 모델 HF **private** 업로드: `psh3333/dacon-bias-qwen25vl`

## 5. 비용 절감 팁
- 학습 끝나면 **pod 즉시 정지/삭제** (시간당 과금)
- 디버깅은 작은 샘플(수십 개)로 먼저 → 파이프라인 검증 후 풀 학습
- 가중치는 RunPod **Network Volume**에 저장하면 pod 재생성 시 재다운로드 불필요

## ⚠️ 규칙 재확인 (제출 전)
- 추론 코드에 API/인터넷 호출 0
- 학습/추론 코드 분리 / UTF-8 / seed 고정 / Private 재현 가능
- 외부·합성 데이터 평가셋 모방 금지(Leakage)
