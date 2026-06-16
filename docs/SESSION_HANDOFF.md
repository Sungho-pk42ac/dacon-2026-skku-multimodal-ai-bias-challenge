# 🔖 세션 핸드오프 (2026-06-15) — 새 세션은 이 문서부터 읽으세요

> DACON 2026 성균관대 멀티모달 AI Bias 챌린지(236722). 솔로, 클라우드 GPU. 목표=대상.
> 같이 볼 것: `docs/H100_MIGRATION.md`(재구축 레시피) · `docs/FINAL-STRATEGY.md`(회의 합의) · `docs/SUBMISSION_GUIDE.md`(2차) · `SCHEDULE.md`(일정) · 프로젝트 메모리.

---

## 1. ⏭️ 즉시 다음 액션 (지금 여기서 이어가기)
**[2026-06-15 09:10 UTC] 새 A6000(Ampere) pod에서 pod_master_unsloth.sh 가동 중.** 모니터링/결과수거가 다음 액션.
1. pod 주소는 로컬 `infra/pod.sh`의 `POD=`에 있음(gitignore, 휘발성). 재시작 시 RunPod 프록시 주소 바뀜 → 새 주소로 갱신.
2. 진행 점검: `bash infra/pod.sh '... > /workspace/x.txt; ...'` 후 패딩 읽기(아래 §10 PTY 주의). master 로그 = `/workspace/master_unsloth.log`.
3. 완료 시 제출 CSV(`submissions/submission_v4.csv`,`_v5.csv`)를 **base64로 pod→로컬 수거**(GitHub 미사용) → 사용자가 DACON 업로드.
4. HF 업로드는 HF_TOKEN 필요(현재 미설정, 학습엔 불필요). W&B는 **offline 모드**(키 주면 sync).
5. A6000이라 v4 SFT ~12h. v4→eval→v5(GRPO)→eval→선택리포트.

## 2. 🔴 현재 라이브 상태
- **A6000(Ampere) 48GB 새 pod**(주소 로컬 `infra/pod.sh`): `pod_master_unsloth.sh` 가동(데이터→SFT v4→eval→제출→GRPO v5→eval→선택리포트). 로그 `/workspace/master_unsloth.log`.
- **환경 검증됨**: lf311(tf5.6/torch2.6+cu124/CUDA✓) · unsloth_env(UNSLOTH_IMPORT_OK) · vllm0.18.
- **코드 전송 = base64 직접전송**(GitHub clone 아님 — RunPod 프록시가 scp/sftp 차단). `/tmp/repo.tgz`(md5 51af6e4a…) 검증 일치. `data_build/test_full.csv`(8500)·`data/placeholder.jpg`(64×64 회색) pod에 존재.
- **W&B=offline**(무인 학습 블로킹 방지). HF_TOKEN 미설정(업로드 단계만 영향).
- ⚠️ 이전 H100 pod는 **셸 백엔드 불량**(인증OK·명령 무응답)으로 폐기, A6000로 회귀. GitHub 배포키 경로는 미사용(전 H100 키만 등록돼 있을 수 있음).

## 3. 결과 / 제출 현황
| 모델 | Public | 상태 |
|---|---|---|
| v2 (Qwen2.5-VL-7B SFT CoT) | **1.0** rank10 | `submissions/submission_v2.csv` 제출됨, HF배포 |
| v3 (Qwen2.5-VL-7B 4ep) | **0.99975** | 제출됨. 속도1.65초→폐기후보 |
| v4 (Qwen3-VL-8B Unsloth SFT 단일토큰) | — | 학습중→H100 재시작. **본명후보** |
| v5 (v4 + Unsloth GRPO) | — | 대기. **본명후보** |
> Public은 운영진 공지상 거의 무의미(오픈벤치). Private(자체제작)·검증방법론·발표가 대상 결정.

## 4. 핵심 전략 (확정)
- **단일토큰 0/1/2** 출력(A/B/C/D 아님 — 실 test가 3지선다 정수라벨). 추론 max_new_tokens=4, 0.5초.
- **Qwen3-VL-8B-Instruct** 베이스(6/1이전·Apache-2.0). Thinking 제외(속도).
- **Unsloth** 학습(SFT cold-start → GRPO). 회원님 train_grpo.ipynb 템플릿.
- **GRPO 목적**: 긴 reasoning 아님. 단일토큰 선택정책 강화(accuracy+format reward).
- **데이터=공개 BBQ만**(외부증강·합성 0 — 누출/실격 회피). make_bbq_clean으로 누출제거+옵션셔플+시나리오dedup+단일토큰.
- **자체검증 중심 선택**: baseline BA(카테고리별)+stress(6순열 위치일관성)+counterfactual(속성swap)+Internal Final Score.

## 5. 사용자(박성호) 결정·선호 (지켜야 함)
- 완전 자율 진행, **확인 안 묻고**(특히 제출). 단 키/비번 타이핑·결제·증명서·DACON업로드는 사용자.
- **DACON 제출은 사용자 수동**(크롬 file_upload 막힘 — 데스크톱 구버전). CSV는 GitHub로 autopush→로컬pull→사용자가 업로드.
- 풀파인튜닝 거부됨(LoRA 유지, 과적합·48GB 이유). 전체 57k×3 thorough 학습 선택.
- 외부데이터 증강 거부(BBQ만). 데이터 철저화는 OK.

## 6. 사용자 역할 (수동 — 나는 못 함)
① pod 대여·결제·SSH제공 ② DACON CSV 업로드·점수회신 ③ HF/W&B 로그인 ④ 재학증명서.

## 7. 규칙 (위반=실격)
오프라인 추론(API 0)·모델 2026-06-01 이전·최종답 LLM생성(0/1/2 토큰)·학습추론 코드분리·0.5초/샘플@A6000·**Data Leakage 금지**(평가셋 모방·합성 금지)·UTF-8. 추론기준환경 A6000 48GB/Py3.10/CUDA12.4/torch2.6.

## 8. 일정 (`SCHEDULE.md`)
06.29 리더보드마감(Private 100%로 상위15팀) → 07.02 2차산출물(코드+발표PDF15분+증명서, dacon@dacon.io) → 07.10 2차평가 → 07.14 시상식(참석필수).

## 9. 핵심 파일 맵
```
prompts.py                         공유: SYSTEM_MESSAGE/USER_TEMPLATE/text_to_label/UNKNOWN_PATTERNS
data_build/make_bbq_clean.py       데이터(셔플/dedup/누출제거/단일토큰)+val_ids SSOT  ← 일관성 원천
data_build/contamination_check.py  train/val↔test.csv 누출검사
data_build/test_full.csv           대회 test 8500 (추론대상)
train/unsloth/sft_unsloth.py       v4 Unsloth SFT (FastVisionModel)
train/unsloth/grpo_unsloth.py      v5 Unsloth GRPO (fast_inference, make_bbq_clean 함수 import)
train/llamafactory/qwen3vl_lora_sft.yaml   (구 LLaMA-Factory, 백업)
inference/baseline_eval.py         정직한 BA+카테고리
inference/stress_eval.py           6순열 위치일관성
inference/counterfactual_stress.py 속성swap 편향측정
inference/make_submission.py       제출 CSV (0/1/2, max_new_tokens=4)
inference/final_model_selection_report.py  Internal Final Score 선택
infra/pod_master_unsloth.sh        Unsloth 풀 오케스트레이터
infra/run_eval_suite.sh            모델별 평가일괄
infra/pod.sh                       SSH헬퍼(gitignore, 휘발성 — H100주소로 갱신)
docs/H100_MIGRATION.md             재구축 레시피
docs/SUBMISSION_GUIDE.md           2차 제출/재현
docs/presentation_outline.md       발표 골격
```

## 10. pod 운영 팁 (시행착오 정리)
- `bash infra/pod.sh '명령'` — 프록시 SSH라 `-tt`+stdin. **출력 첫 줄 자주 잘림 → 더미 echo 먼저**.
- 긴 작업은 `setsid bash ... >log 2>&1 </dev/null &` (PC 꺼도 진행).
- pod git: `git config core.sshCommand "ssh -i ~/.ssh/gh_pod -o IdentitiesOnly=yes"` (배포키, 쓰기가능).
- 진행로그 볼 때 `tr "\r" "\n" < log` (tqdm \r 때문).
- 3 venv: unsloth_env(학습)·lf311(데이터+추론)·grpo3_env(구TRL백업).

## 11. 미완료 체크리스트
- [ ] H100 전환 + v4 Unsloth SFT 완주 → 정직한 BA·0.5초실측·위치일관성
- [ ] v5 Unsloth GRPO → v4와 비교 → final_model_selection_report로 최종선택
- [ ] 최종 submission CSV DACON 업로드(사용자)
- [ ] 2차: 발표 PDF(검증방법론 서사 = 운영진 정조준) + 재학증명서
