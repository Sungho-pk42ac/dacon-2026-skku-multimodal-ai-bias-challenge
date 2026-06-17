"""
외부검증 리포트 생성기 (집계 전용, 모델 추론 없음).
입력:
  outputs/eval_results_v4_v8.json        — v8_eval(900) 인분포+OOD 다지표 (base/llava/v6/v7/v8a/v8b)
  outputs/external_validation_results.json — ood2(600) 독립 일반화 + 이미지 ablation (base/v6/v8b...)
출력:
  outputs/external_validation_table.csv   — 모델×지표 통합 비교표
  outputs/external_validation_ablation.json — 이미지(placeholder vs blank)·옵션셔플 ablation 요약
  docs/EXTERNAL_VALIDATION_REPORT.md      — 서술형 리포트(표 포함)

사용: python inference/gen_external_reports.py --root .
"""
import argparse, json, os

V8_KEYS = ["balanced_acc", "ood_acc", "shuffle_consistency", "s_AMB", "s_DIS",
           "output_format_validity", "inference_speed_sec_per_sample"]
ORDER = ["base", "llava_ov_7b", "v6", "v7", "v8a", "v8b"]


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    R = args.root
    v8 = load(os.path.join(R, "outputs/eval_results_v4_v8.json"))
    ext = load(os.path.join(R, "outputs/external_validation_results.json")).get("ood2", {})

    tags = [t for t in ORDER if t in v8 or t in ext] + \
           [t for t in set(list(v8) + list(ext)) if t not in ORDER]

    # ── 통합 표 CSV ──────────────────────────────────────
    csv_path = os.path.join(R, "outputs/external_validation_table.csv")
    rows = []
    header = ["metric"] + tags
    rows.append(header)
    # v8_eval(인분포) 지표
    for k in V8_KEYS:
        rows.append([f"v8eval_{k}"] + [v8.get(t, {}).get(k, "") for t in tags])
    # ood2(독립) 지표
    for k in ["accuracy", "shuffle_consistency", "image_invariance_ph_vs_blank",
              "output_format_validity", "inference_speed_sec_per_sample"]:
        rows.append([f"ood2_{k}"] + [ext.get(t, {}).get(k, "") for t in tags])
    with open(csv_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # ── ablation JSON ────────────────────────────────────
    abl = {"image_ablation_placeholder_vs_blank": {}, "option_shuffle": {}, "note":
           "ood2는 텍스트전용 MC셋 → 이미지는 placeholder가 표준. blank(흰배경) 대비 invariance로 "
           "'무관 이미지에 답이 흔들리지 않는가' 측정. real/shuffled-image ablation은 이미지 데이터셋 "
           "(A-OKVQA 등) 필요 — 이번 패스 미수행(사유: pod 다운로드 제약, 본 검증은 텍스트 일반화 집중)."}
    for t in tags:
        if t in ext:
            abl["image_ablation_placeholder_vs_blank"][t] = ext[t].get("image_invariance_ph_vs_blank")
            abl["option_shuffle"][t] = {"ood2_shuffle_consistency": ext[t].get("shuffle_consistency"),
                                         "v8eval_shuffle_consistency": v8.get(t, {}).get("shuffle_consistency")}
    json.dump(abl, open(os.path.join(R, "outputs/external_validation_ablation.json"), "w"),
              ensure_ascii=False, indent=2)

    # ── 서술형 리포트 ────────────────────────────────────
    def cell(t, src, k):
        v = (ext if src == "ood2" else v8).get(t, {}).get(k, "-")
        return "-" if v == "" or v is None else v
    md = os.path.join(R, "docs/EXTERNAL_VALIDATION_REPORT.md")
    finetuned = [t for t in tags if t not in ("base", "llava_ov_7b")]
    with open(md, "w", encoding="utf-8") as f:
        f.write("# 외부검증 리포트 — v6 vs v8A vs v8B (독립 공개셋)\n\n")
        f.write("> eval-only. 학습/파인튜닝 없음. label은 모델 생성 텍스트에서만 파싱. 외부 API 0.\n")
        f.write("> 인분포(v8_eval 900: BBQ+혼합OOD) + **독립 held-out ood2(600: MMLU/HellaSwag/ARC-C/WinoGrande)**.\n")
        f.write("> ood2는 어떤 버전도 학습에 쓰지 않은 순수 미관측 셋. 텍스트전용이라 이미지=placeholder, blank invariance ablation 포함.\n\n")
        f.write("## 통합 비교표\n\n")
        cols = tags
        f.write("| 지표 | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        named = [("v8eval balanced_acc", "v8", "balanced_acc"),
                 ("v8eval OOD acc", "v8", "ood_acc"),
                 ("v8eval shuffle 일관성", "v8", "shuffle_consistency"),
                 ("v8eval s_AMB", "v8", "s_AMB"), ("v8eval s_DIS", "v8", "s_DIS"),
                 ("v8eval fmt유효", "v8", "output_format_validity"),
                 ("v8eval 속도(s)", "v8", "inference_speed_sec_per_sample"),
                 ("**ood2 정확도(독립)**", "ood2", "accuracy"),
                 ("ood2 shuffle 일관성", "ood2", "shuffle_consistency"),
                 ("ood2 image-invariance(ph/blank)", "ood2", "image_invariance_ph_vs_blank"),
                 ("ood2 fmt유효", "ood2", "output_format_validity"),
                 ("ood2 속도(s)", "ood2", "inference_speed_sec_per_sample")]
        for label, src, k in named:
            f.write(f"| {label} | " + " | ".join(str(cell(t, src, k)) for t in cols) + " |\n")
        f.write("\n## ood2 소스별 정확도\n\n")
        srcs = sorted({s for t in ext for s in ext[t].get("accuracy_by_src", {})})
        if srcs:
            f.write("| 모델 | " + " | ".join(srcs) + " |\n|" + "---|" * (len(srcs) + 1) + "\n")
            for t in tags:
                if t in ext:
                    f.write(f"| {t} | " + " | ".join(str(ext[t].get("accuracy_by_src", {}).get(s, "-")) for s in srcs) + " |\n")
        f.write("\n## 판정 요약(자동)\n\n")
        f.write("- 파인튜닝 모델: " + ", ".join(finetuned) + "\n")
        f.write("- base/LLaVA(비파인튜닝)와의 격차는 본문 표 참조(편향과제 fmt유효·balanced에서 큰 차이 예상).\n")
        f.write("- 최종 결정 규칙과 권고는 docs/FINAL_MODEL_SELECTION.md 참조.\n")
    print("GEN_REPORTS_DONE", csv_path, md, flush=True)


if __name__ == "__main__":
    main()
