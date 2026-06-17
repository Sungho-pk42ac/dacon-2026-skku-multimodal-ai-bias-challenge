# v4 → v8B 전체 진행·비교 정리 (완본)

> DACON 2026 성균관대 멀티모달 AI Bias 챌린지(236722). base **Qwen3-VL-8B-Instruct** (Apache-2.0).
> 모든 정량치는 **동일 held-out 공개셋 `v8_eval`(900, BBQ+OOD)** 에서 transformers 단일샘플/배치로 측정.
> 누출방지: DACON 평가셋·테스트 미사용, val_ids + v8_eval 학습 제외, sha1 sample_id 중복/누출 체크.
> 최종 라벨은 항상 **모델 생성 텍스트 파싱**(규칙기반 선택 아님). 외부 API 추론 0.

---

## 1. 한눈에 보는 전체 비교표

| 지표 | v4 | v5 | v6 | v7 | v8A | **v8B** |
|---|---|---|---|---|---|---|
| 학습 방식 | 단일토큰 SFT | v4+GRPO(BBQ전체) | v4+GRPO(하드네거) | **콜드스타트SFT+추론GRPO** | v4+강건성GRPO(rank32) | **v4+강건성GRPO(rank16)** |
| 콜드스타트 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| BBQ ambiguous acc | 1.0 | ≈1.0 | 1.0 | **0.7067** | 1.0 | **1.0** |
| BBQ disambiguated acc | 1.0 | ≈1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| **OOD acc** | 0.8033 | ≈0.803 | 0.8083 | **0.6733** | **0.8117** | **0.8067** |
| **option-shuffle consistency** | 0.9478 | ≈0.948 | 0.9456 | **0.7944** | **0.9411** | **0.9456** |
| unknown-position consistency | 1.0 | — | 1.0 | — | 1.0 | **1.0** |
| ambiguous person-selection error | 0.0 | — | 0.0 | **0.2933** | 0.0 | **0.0** |
| disambiguated over-abstain | 0.0 | — | 0.0 | 0.0 | 0.0 | **0.0** |
| output format validity | 0.9556 | — | 0.9533 | — | 0.9733 | **0.9544** |
| inference speed (s/sample) | 0.0586 | — | 0.0597 | — | 0.0584 | **0.0582** |
| bias s_AMB / s_DIS | 0/0.04 | — | 0/0.04 | **-0.0667/0.04** | 0/0.04 | **0/0.04** |
| 분류/판정 | 기준선 | 효과0 | 안정 | **FAIL(붕괴)** | OOD↑·셔플↓ | **PASS** |
| HF 배포 | v4-unsloth | v5-unsloth | v6 | v7 | v8a | **v8b** |

*v5는 v8_eval 재측정 미실시(이전 측정상 v4와 동급, GRPO 효과 0이라 ≈ 표기). v7의 수치는 v8_eval(혼합셋) 기준이며,
별도 OOD-only 800셋에서는 0.416까지 떨어짐(`V4_V7_SUMMARY.md`).*

### OOD 소스별 (v4 / v8A / v8B)
| 소스 | v4 | v8A | v8B |
|---|---|---|---|
| ood_arc | 0.9533 | 0.9533 | 0.9533 |
| ood_csqa | 0.74 | 0.76 | 0.74 |
| ood_obqa | 0.76 | 0.7533 | 0.76 |
| ood_siqa | 0.76 | 0.78 | **0.7733** |

---

## 2. 버전별 진행 서사 (무엇을·왜·결과)

### v4 — 기준선 (단일토큰 SFT)
- 공개 BBQ(셔플·dedup·누출제거)로 가벼운 SFT. 0/1/2 직답.
- **가장 단순·강건.** OOD 0.8033, 셔플 0.9478. 이후 모든 실험의 base.

### v5 — v4 + 단일토큰 GRPO (BBQ 전체)
- 가설: GRPO로 BBQ 보강. **결과: 효과 0** — BBQ val 포화로 보상 분산≈0 → v4와 동급.
- 교훈: 포화된 분포엔 GRPO 신호가 없다.

### v6 — v4 + 단일토큰 GRPO (하드네거티브)
- v4 오답 하드샘플 위주 GRPO. OOD 0.8083(미세↑), 셔플 0.9456. v4와 동급·안정. 빠름.
- "GRPO 자체는 일반화를 해치지 않는다"의 증거(콜드스타트 없을 때).

### v7 — 콜드스타트 SFT + 추론 GRPO ❌ **실패**
- BBQ 100% 콜드스타트 + "정답 박은 추론 템플릿" + rank64.
- **전방위 붕괴**: amb 0.7067, OOD 0.6733, 셔플 0.7944, 사람선택오류 0.2933, s_AMB 음수(역편향).
- ablation으로 범인 확정: 콜드스타트 SFT만으로 이미 OOD 0.42(GRPO 무죄).
- = Chu 2025 "SFT Memorizes, RL Generalizes" 예측 그대로. **"특화·암기할수록 일반화 붕괴."**

### v8A — 강건성 GRPO 1차 (rank32)
- v4 base, 콜드스타트 제거, 강건성 보상 도입. **OOD 0.8117로 최고치** 달성.
- 그러나 **셔플 일관성 0.9411로 하락** — OOD↑를 옵션순서 강건성↓와 맞바꾼 trade-off.
- format 보상이 죽은 신호(−0.5 고정) 버그 존재(학습엔 무해하나 비효율).

### v8B — 강건성 GRPO 2차 (rank16) ✅ **PASS**
- **목표: v8A의 trade-off 해소** — OOD 유지하며 셔플 일관성 회복.
- 처방: ① 저rank(16)·저lr(1e-6)로 v4를 덜 흔듦 ② **셔플짝 데이터증강**(원본+셔플 2행) ③ shuffle-consistency 보상(정답텍스트 순서무관 일치) ④ source-normalized 보상(소스별 zero-centered) ⑤ format 버그 수정 ⑥ 동적샘플링(useful 위주).
- 풀: v4·v8A 채점으로 채굴한 OOD 셔플불일치 91 + 양쪽오답 120 + 회귀 7 + OOD이득 2 → 170고유×셔플짝=340행. (BBQ는 v4·v8A 완벽이라 동적샘플링상 too-easy로 자동 제외 → OOD 집중)
- **결과: 셔플 0.9411→0.9456 회복**(v6 수준), OOD 0.8067(v4↑·v8A−0.005), BBQ 완벽·과기권0·사람오류0·format안정.
- 판정 **PASS** (STRONG 미달: OOD<0.8150·셔플<0.9470).

---

## 3. 성장 분석 — 무엇이 일반화를 좌우했나

```
                       OOD      셔플일관성   해석
v4  (가벼운 SFT)       0.8033   0.9478      가장 강건한 기준선
v5  (+GRPO 포화)       ~0.803   ~0.948      효과 0 (분산 없음)
v6  (+GRPO 하드)       0.8083   0.9456      안정 유지
v7  (콜드스타트+추론)  0.6733   0.7944      붕괴 (특화 과적합·망각)
v8A (강건성 GRPO r32)  0.8117   0.9411      OOD↑ ↔ 셔플↓ trade-off
v8B (강건성 GRPO r16)  0.8067   0.9456      trade-off 해소 (PASS)
```

**핵심 결론 3가지**
1. **콜드스타트 SFT가 일반화의 적**(v7). RL(GRPO) 단독은 일반화를 해치지 않음(v6/v8A/v8B). → Chu 2025 부합.
2. **GRPO로 얻는 향상폭은 노이즈 수준(±수문항)** — base(v4)의 잠재능력 한계 내. BBQ는 이미 천장(1.0)이라 변별 불가.
3. **v8B의 기여는 "방법론"** — 강건성 보상·셔플짝·소스정규화로 *v8A의 회귀를 되돌릴 수 있음*을 입증. 점수보다 **검증 서사**가 차별점.

---

## 4. 제출 권고
- **메인 후보: v8B** (사용자 지정 main candidate). GRPO-only("v4 단독 아님" 충족), PASS, 셔플 회복, BBQ 완벽, 빠름, 규칙 준수. base Apache-2.0.
- 대안: v6(동급, 더 단순). v8A(OOD 최고이나 셔플 약점). **v4/v7은 제출 안 함**(v4=단독SFT, v7=실패).
- 상세: `FINAL_V8B_REPORT.md`, `FINAL_MODEL_SELECTION.md`.

---

## 5. 산출물 위치
- 모델(HF): `psh3333/dacon-skku-bias-vlm-{v4-unsloth,v5-unsloth,v6,v7,v8a,v8b}`
- W&B: `psh243360-korea-university/dacon-bias/runs/xm3p2p81` (run `v8B_robust_grpo`)
- 평가: `outputs/eval_results_v4_v8.json`, `outputs/eval_results_v8b.json`, `outputs/v8b_comparison_table.csv`
- 리포트: `docs/{FINAL_V8B_REPORT,V8B_EVAL_REPORT,V8B_POOL_REPORT,V8B_WANDB_SUMMARY,V4_V7_SUMMARY,DACON_RULE_COMPLIANCE_CHECKLIST}.md`
- 제출: `submissions/submission_v8b.csv`(+v8a) — 실제 테스트 이미지 확보 후 생성, `*_validation.json`로 검증.
