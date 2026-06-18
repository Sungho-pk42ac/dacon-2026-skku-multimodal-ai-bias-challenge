"""W&B 외부검증 로깅 (run: external_validation_v6_v8a_v8b). 추론 없음, JSON만 업로드."""
import json, os
import wandb

R = "/workspace/dacon-bias-challenge"
v8 = json.load(open(f"{R}/outputs/eval_results_v4_v8.json", encoding="utf-8"))
ext = json.load(open(f"{R}/outputs/external_validation_results.json", encoding="utf-8"))["ood2"]

run = wandb.init(project="dacon-skku-bias", name="external_validation_v6_v8a_v8b",
                 config={"type": "external_validation", "train": False,
                         "indist_set": "v8_eval(900)", "independent_set": "ood2(600) MMLU/HellaSwag/ARC-C/WinoGrande"})

cols = ["model", "v8eval_balanced", "v8eval_ood", "v8eval_shuffle", "v8eval_s_AMB",
        "v8eval_fmt", "v8eval_speed", "ood2_acc", "ood2_shuffle", "ood2_invariance", "ood2_fmt"]
tbl = wandb.Table(columns=cols)
for t in ["base", "llava_ov_7b", "v6", "v7", "v8a", "v8b"]:
    a = v8.get(t, {})
    b = ext.get(t, {})
    tbl.add_data(t, a.get("balanced_acc"), a.get("ood_acc"), a.get("shuffle_consistency"),
                 a.get("s_AMB"), a.get("output_format_validity"), a.get("inference_speed_sec_per_sample"),
                 b.get("accuracy"), b.get("shuffle_consistency"),
                 b.get("image_invariance_ph_vs_blank"), b.get("output_format_validity"))
run.log({"external_validation_table": tbl})
run.summary.update({
    "decision": "v6", "rule": "tied->v6; v8a/v8b do not clearly beat v6",
    "v6_ood2_acc": ext["v6"]["accuracy"], "v8b_ood2_acc": ext["v8b"]["accuracy"], "base_ood2_acc": ext["base"]["accuracy"],
    "v6_v8eval_ood": v8["v6"]["ood_acc"], "v6_v8eval_balanced": v8["v6"]["balanced_acc"],
})
run.finish()
print("WANDB_EXTERNAL_DONE", run.url if hasattr(run, "url") else "ok", flush=True)
