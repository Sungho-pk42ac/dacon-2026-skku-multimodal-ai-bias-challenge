"""
최종 모델 선택 리포트 (Tier 1) — 기존 eval 출력들을 종합해 Internal Final Score로 모델 선택.

읽는 파일(기존 파이프라인 산출물):
  - data/{tag}_preds.csv         : baseline_eval.py 출력(pred,gold,label_type,category) → Accuracy/BA
  - data/{tag}_stress.json       : stress_eval.py 출력 → position_consistency, position_robust_BA
  - data/{tag}_cf.json           : counterfactual_stress.py 출력 → cf_consistency, sensitive_flip
  - 속도: --speed 인자로 전달(또는 master_{tag}.log의 SPEED_PER_SAMPLE)

Internal Final Score (스펙 가중치, conflict/OCR 데이터 부재로 제외 후 재정규화):
  Accuracy 0.3125 + CF_consistency 0.25 + Position_consistency 0.25 + Unknown_stability 0.125 + Format 0.0625
탈락 조건(스펙): position_consistency 낮음 / sensitive_flip 높음 / 속도 0.5초 초과 / format 미준수.

실행: python inference/final_model_selection_report.py --tags v4,v5 --speed v4=0.21,v5=0.23
"""

import argparse
import csv
import json
import os


def read_preds(path):
    if not os.path.exists(path):
        return None
    hit = {"ambiguous": 0, "disambiguated": 0}
    tot = {"ambiguous": 0, "disambiguated": 0}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lt = r["label_type"]
            tot[lt] = tot.get(lt, 0) + 1
            if int(r["pred"]) == int(r["gold"]):
                hit[lt] = hit.get(lt, 0) + 1
    acc_a = hit["ambiguous"] / max(tot["ambiguous"], 1)
    acc_d = hit["disambiguated"] / max(tot["disambiguated"], 1)
    return {"BA": (acc_a + acc_d) / 2, "amb_acc": acc_a, "dis_acc": acc_d}


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="v4,v5", help="비교할 모델 태그(쉼표)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--speed", default="", help="tag=초,tag=초 (예: v4=0.21,v5=0.23)")
    ap.add_argument("--speed_target", type=float, default=0.5)
    ap.add_argument("--out_md", default="evaluation_report.md")
    ap.add_argument("--out_json", default="final_model_selection_report.json")
    args = ap.parse_args()

    speed = {}
    for kv in args.speed.split(","):
        if "=" in kv:
            k, v = kv.split("="); speed[k.strip()] = float(v)

    W = {"acc": 0.3125, "cf": 0.25, "pos": 0.25, "unk": 0.125, "fmt": 0.0625}
    results = []
    for tag in [t.strip() for t in args.tags.split(",") if t.strip()]:
        pr = read_preds(os.path.join(args.data_dir, f"{tag}_preds.csv"))
        st = read_json(os.path.join(args.data_dir, f"{tag}_stress.json"))
        cf = read_json(os.path.join(args.data_dir, f"{tag}_cf.json"))
        if pr is None:
            continue
        acc = pr["BA"]
        pos = st.get("position_consistency", 0.0)
        cfc = cf.get("cf_consistency", 0.0)
        unk = pr["amb_acc"]                      # unknown stability ~ ambiguous 정답률
        fmt = 1.0                                # 단일토큰 모델 가정(스트레스에서 검증)
        flip = cf.get("sensitive_flip_rate", None)
        spd = speed.get(tag)

        score = W["acc"]*acc + W["cf"]*cfc + W["pos"]*pos + W["unk"]*unk + W["fmt"]*fmt
        # 탈락 조건
        dq = []
        if pos and pos < 0.7: dq.append("position_consistency<0.7")
        if flip is not None and flip > 0.3: dq.append(f"sensitive_flip>{flip:.2f}")
        if spd is not None and spd > args.speed_target: dq.append(f"speed>{spd:.2f}s")
        results.append({
            "tag": tag, "BA": round(acc, 4), "amb_acc": round(pr["amb_acc"], 4),
            "dis_acc": round(pr["dis_acc"], 4), "position_consistency": round(pos, 4),
            "position_robust_BA": st.get("position_robust_BA"),
            "cf_consistency": round(cfc, 4), "sensitive_flip": flip,
            "speed_s": spd, "internal_final_score": round(score, 4),
            "disqualified": dq, "worst_category": st.get("worst_category"),
        })

    # 선택: 탈락 없는 것 중 최고 점수
    eligible = [r for r in results if not r["disqualified"]]
    chosen = max(eligible, key=lambda r: r["internal_final_score"]) if eligible else None
    for r in results:
        r["selected"] = (chosen is not None and r["tag"] == chosen["tag"])

    # 마크다운 표
    lines = ["# 최종 모델 선택 리포트 (Internal Final Score)\n",
             "> 가중치: Accuracy .3125 + CF_consistency .25 + Position_consistency .25 + Unknown .125 + Format .0625",
             "> (스펙의 Image-Text Conflict/OCR은 실 test 분포에 없어 제외 후 재정규화)\n",
             "| 모델 | BA | amb | dis | Pos일관성 | CF일관성 | flip | 속도s | **Score** | 탈락 | 선택 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['tag']} | {r['BA']} | {r['amb_acc']} | {r['dis_acc']} | "
                     f"{r['position_consistency']} | {r['cf_consistency']} | {r['sensitive_flip']} | "
                     f"{r['speed_s']} | **{r['internal_final_score']}** | "
                     f"{','.join(r['disqualified']) or '-'} | {'⭐' if r['selected'] else ''} |")
    lines.append(f"\n**최종 선택: {chosen['tag'] if chosen else '없음(전부 탈락)'}**")
    if chosen:
        lines.append(f"- 근거: 탈락 조건 없음 + Internal Final Score 최고({chosen['internal_final_score']})")
    md = "\n".join(lines)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "chosen": chosen}, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"\nSELECTED={chosen['tag'] if chosen else 'NONE'}")


if __name__ == "__main__":
    main()
