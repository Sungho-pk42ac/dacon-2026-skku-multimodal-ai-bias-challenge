# H100 이전 / 프로젝트 상태 기록 (2026-06-15)

> A6000(12.5h) → H100(~3-4h)로 학습 가속 전환. 이 문서 하나로 H100에서 전 파이프라인 재구축 가능.

---

## 0. 한 줄 요약
공개 **walledai/BBQ**를 누출제거·옵션셔플·단일토큰으로 정제 → **Qwen3-VL-8B-Instruct**를 **Unsloth**로 SFT(v4)+GRPO(v5) → 자체 검증(BA·위치일관성·counterfactual)으로 모델 선택 → 단일토큰 0/1/2 오프라인 추론으로 제출.

## 1. 현재까지 결과
| 모델 | 방법 | Public | 비고 |
|---|---|---|---|
| v2 | Qwen2.5-VL-7B SFT(CoT) | **1.0** (rank 10) | `submissions/submission_v2.csv`, HF 배포됨 |
| v3 | Qwen2.5-VL-7B SFT(4ep) | **0.99975** | 속도 1.65초(위반)→폐기후보. `submission_v3.csv` |
| **v4** | **Qwen3-VL-8B Unsloth SFT 단일토큰** | (학습중→H100 재시작) | 본명 후보 |
| **v5** | **v4 위 Unsloth GRPO** | (대기) | 본명 후보 |

⚠️ **Public은 운영진 공지상 거의 무의미**(오픈벤치 샘플). Private=자체제작 → 일반화·검증방법론이 핵심.

## 2. 베이스 모델
- **Qwen/Qwen3-VL-8B-Instruct** (2025-09-23 공개=6/1이전✅, Apache-2.0). Thinking 아님(속도).

## 3. 환경 (3 venv)
```bash
# (A) lf311 — 데이터 생성 + 추론(transformers 5.6, Qwen3-VL merged 로드)
uv venv --python 3.11 /workspace/lf311 && source /workspace/lf311/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers==5.6.0" accelerate peft "datasets>=3" qwen-vl-utils pillow pandas
git clone https://github.com/hiyouga/LLaMA-Factory /workspace/LLaMA-Factory && uv pip install -e /workspace/LLaMA-Factory
uv pip install hf-transfer "huggingface_hub[cli]" openai wandb

# (B) unsloth_env — 학습(SFT+GRPO). 회원님 train_grpo.ipynb 템플릿
uv venv --python 3.11 /workspace/unsloth_env && source /workspace/unsloth_env/bin/activate
uv pip install unsloth wandb typing_extensions
uv pip install "vllm==0.18.0"
# 검증: python -c "from unsloth import FastVisionModel; print('ok')"
```
- HF 로그인: `hf auth login` (user psh3333) · W&B: `wandb login` — **사용자가 직접**(키 타이핑 불가).

## 4. 데이터 파이프라인 (일관성 핵심)
`data_build/make_bbq_clean.py` — walledai/BBQ → :
- sha1 `sample_id` + 중복제거 + 품질필터(unknown 1개)
- **시나리오 단위 dedup**(질문+선택지) → train/val 누출 0
- **옵션 셔플**(라벨 재매핑) → 위치편향 제거 (정답분포 0/1/2 균등 확인됨)
- **단일토큰 타깃** "0"/"1"/"2"
- 출력: `val_ids.json`(SSOT) · `bbq_val_clean.jsonl`(1320, amb/dis 균형) · `bbq_v4_train.json`(57094)
> GRPO도 `make_bbq_clean`의 `sample_id/shuffle_opts/is_unknown`를 **import** → v4·v5 완전 동일 처리.

## 5. 학습 스크립트 (Unsloth)
- `train/unsloth/sft_unsloth.py` — FastVisionModel, bbq_v4_train.json 재사용, vision층 동결+언어 LoRA r32, 단일토큰. `save_pretrained_merged(merged_16bit)`.
- `train/unsloth/grpo_unsloth.py` — FastVisionModel `fast_inference=True`, val_ids 배제+옵션셔플, reward=accuracy+format(0/1/2). `UNSLOTH_VLLM_STANDBY=1`.

## 6. 평가/추론 (lf311, transformers)
- `inference/baseline_eval.py` — 정직한 BA + 카테고리별
- `inference/stress_eval.py` — 6순열 위치 강건성/일관성
- `inference/counterfactual_stress.py` — 사회속성 swap → cf_consistency/sensitive_flip
- `inference/make_submission.py` — 0/1/2 단일토큰 제출 CSV (max_new_tokens=4)
- `inference/final_model_selection_report.py` — Internal Final Score로 v4/v5 선택
- `data_build/contamination_check.py` — train/val ↔ test.csv 누출검사
- `infra/run_eval_suite.sh <tag> <model>` — 위 평가 일괄

## 7. 오케스트레이터
`infra/pod_master_unsloth.sh` — 데이터→SFT(v4)→eval→submission→GRPO(v5)→eval→submission→선택리포트→HF.
실행: `setsid bash infra/pod_master_unsloth.sh >/workspace/master_unsloth.log 2>&1 </dev/null &`

## 8. 인프라 헬퍼
- `infra/pod.sh` (gitignore, 휘발성) — SSH 원격실행. **H100 새 주소로 `POST=...` 갱신 필요.** 프록시면 `-tt`+stdin.
- autopush: submission CSV → GitHub 자동 push (배포키 gh_pod, `git config core.sshCommand "ssh -i ~/.ssh/gh_pod -o IdentitiesOnly=yes"`).
- test_full.csv(대회 test 8500) = `data_build/test_full.csv` (repo에 있음).

## 9. H100 재구축 순서
1. H100 pod 대여(RunPod), passwordless SSH 확보 → `infra/pod.sh`의 POD 주소 갱신.
2. 배포키 등록 → `git clone` repo → `cd dacon-bias-challenge`.
3. §3 환경(lf311 + unsloth_env) 설치.
4. `hf auth login` + `wandb login` (사용자).
5. autopush 워커 재기동(`infra/autopush_submissions.sh` 패턴).
6. `setsid bash infra/pod_master_unsloth.sh ...` 런치.
> H100이면 v4 SFT ~3-4h. 0.5초 속도 게이트는 **최종 추론만 A6000급에서 재검증**(또는 단일토큰이라 여유).

## 10. 미완료 (H100에서 할 것)
- [ ] v4 Unsloth SFT 완주 → 정직한 BA + 0.5초 실측 + 위치일관성
- [ ] v5 Unsloth GRPO → v4와 비교
- [ ] final_model_selection_report로 최종 모델 선택
- [ ] 2차: 발표 PDF(검증방법론 서사) + 재학증명서

## 11. 규칙 준수 (확정)
오프라인 추론·모델6/1이전·학습추론분리·단일토큰 LLM생성답·BBQ만(외부증강 0)·누출제거·UTF-8. (`SUBMISSION_GUIDE.md` 참조)
