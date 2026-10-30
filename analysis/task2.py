"""Task 2 report assets: learning curves with stage boundaries, final tables (both suites, 4 conditions),
correct-image advantage with pair-bootstrap CIs, resources, 196 vs 49 tokens, and aligned qualitative examples.

python -m analysis.task2
"""
import json

import numpy as np

from analysis.common import INK2, NEUTRAL, SLOTS, TAB, jd, jl, plt, save, write_table

R = "results/task2"
CONDS = ["correct", "complementary", "mismatched", "noimage"]
CNAME = {"correct": "Correct", "complementary": "Complementary swap", "mismatched": "Mismatched", "noimage": "No image"}
ENDS = [2250, 3750]
CCOL = {"correct": SLOTS[0], "complementary": SLOTS[1], "noimage": NEUTRAL}


def stage_lines(ax, top=True):
    for e in ENDS:
        ax.axvline(e, color=NEUTRAL, lw=0.7, ls=":")
    if top:
        y = ax.get_ylim()[1]
        for x, t in [(1125, "S1 caption"), (3000, "S2 instruct"), (4125, "S3 VQA")]:
            ax.text(x, y, t, ha="center", va="bottom", fontsize=6, color=INK2)


def curves(run="main"):
    tl = jl(f"{R}/{run}/train_log.jsonl")
    dv = jl(f"{R}/{run}/dev_log.jsonl")
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 3.9))
    u = np.array([x["update"] for x in tl])
    l = np.array([x["loss"] for x in tl])
    k = 50
    sm = np.convolve(l, np.ones(k) / k, mode="valid")
    ax = axs[0, 0]
    ax.plot(u, l, color=SLOTS[0], lw=0.3, alpha=0.25)
    ax.plot(u[k - 1:], sm, color=SLOTS[0], lw=1.3)
    ax.set_ylim(0, 5)
    ax.set_title("(a) training loss (mean over 50 updates)", pad=10)
    ax.set_ylabel("example-weighted CE")
    stage_lines(ax)
    du = [d["update"] for d in dv]
    ax = axs[0, 1]
    ax.plot(du, [d["caption_loss"] for d in dv], color=SLOTS[0], marker="o", ms=3)
    ax.set_title("(b) dev caption reference loss", pad=10)
    ax.set_ylabel("CE per token")
    stage_lines(ax)
    ax = axs[1, 0]
    for suite, ls, mk in [("grounding_dev", "-", "o"), ("vqa_dev", "--", "s")]:
        for c in ["correct", "complementary", "noimage"]:
            ax.plot(du, [d[suite][f"nll_{c}"] for d in dv], color=CCOL[c], ls=ls, marker=mk, ms=2.5, lw=1.1,
                    label=f"{'grounding' if suite[0] == 'g' else 'VQAv2'}, {CNAME[c].lower()}")
    ax.set_ylim(0, 8)
    ax.set_title("(c) dev reference-answer NLL by image input")
    ax.set_ylabel("NLL (no termination)")
    ax.set_xlabel("update")
    stage_lines(ax, top=False)
    hc, lc = ax.get_legend_handles_labels()
    ax = axs[1, 1]
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.plot(du, [d["grounding_dev"]["swap_penalty"] for d in dv], color=SLOTS[2], marker="o", ms=3, label="grounding")
    ax.plot(du, [d["vqa_dev"]["swap_penalty"] for d in dv], color=SLOTS[6], ls="--", marker="s", ms=3, label="VQAv2")
    ax.set_title(r"(d) image-swap penalty $\Delta$NLL")
    ax.set_ylabel(r"NLL(partner) $-$ NLL(correct)")
    ax.set_xlabel("update")
    ax.legend(loc="upper left")
    stage_lines(ax, top=False)
    fig.tight_layout()
    fig.legend(hc, lc, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=6.3, title="(c): suite, image input (solid = grounding, dashed = VQAv2)", title_fontsize=6.3)
    save(fig, f"t2_curves_{run}")
    return dv


def final_tables(run="main"):
    e = jd(f"{R}/{run}/final_eval.json")
    out = {}
    for suite in ["grounding", "vqa"]:
        S = e["suites"][suite]
        types = list(S["correct"]["by_type"])
        rows = []
        for c in CONDS:
            s = S[c]
            r = {"Image input": CNAME[c], "Accuracy": 100 * s["accuracy"], "95% CI": "[{:.1f}, {:.1f}]".format(*[100 * x for x in s["accuracy_ci95"]])}
            for t in types:
                r[t] = 100 * s["by_type"][t]
            r.update({"Pair acc.": 100 * s["pair_accuracy"], "Identical": 100 * s["identical_answer_rate"], "Empty": 100 * s["empty_answer_rate"],
                      "Follows partner": 100 * s["follows_partner_answer_rate"], "Ref. NLL": s["mean_ref_nll"], "ms/answer": 1000 * s["gen_time_per_answer_s"]})
            rows.append(r)
        cols = ["Image input", "Accuracy", "95% CI"] + types + ["Pair acc.", "Identical", "Empty", "Follows partner", "Ref. NLL", "ms/answer"]
        f1 = lambda v: f"{v:.1f}"
        write_table(rows, cols, f"t2_final_{suite}_{run}", {c: (lambda v: f"{v:.2f}") if c == "Ref. NLL" else f1 for c in cols[1:] if c != "95% CI"})
        out[suite] = S["advantage_correct_vs_complementary"]
    adv = [{"Suite": "Grounding" if s == "grounding" else "VQAv2", "Acc. advantage (pp)": a["accuracy_pp"], "Acc. 95% CI": "[{:.1f}, {:.1f}]".format(*a["accuracy_pp_ci95"]),
            "Swap NLL penalty": a["nll_swap_penalty"], "NLL 95% CI": "[{:.2f}, {:.2f}]".format(*a["nll_swap_penalty_ci95"]),
            **{f"{t} (pp)": v for t, v in a["by_type_accuracy_pp"].items()}} for s, a in out.items()]
    cols = sorted({k for r in adv for k in r}, key=lambda k: (k not in ("Suite", "Acc. advantage (pp)", "Acc. 95% CI", "Swap NLL penalty", "NLL 95% CI"), k))
    write_table(adv, cols, f"t2_advantage_{run}", {"Acc. advantage (pp)": lambda v: f"{v:+.1f}", "Swap NLL penalty": lambda v: f"{v:+.2f}",
                                                 **{c: (lambda v: f"{v:+.1f}") for c in cols if c.endswith("(pp)") and c != "Acc. advantage (pp)"}})
    return e


def resources():
    fc = jd(f"{R}/feature_cache.json")
    rows = []
    for run in ["main", "pool49"]:
        t = jd(f"{R}/{run}/train_summary.json")
        e = jd(f"{R}/{run}/final_eval.json")
        for st in ["caption_alignment", "visual_instruction", "natural_vqa"]:
            tp = t["trainable_params"][st]
            rows.append({"Run": f"{t['n_visual_tokens']} tokens", "Stage": st.replace("_", " "), "Trainable (M)": tp["total"] / 1e6,
                         "Connector (M)": tp["connector"] / 1e6, "LoRA (M)": tp["lora"] / 1e6, "Time (min)": t["stage_time_s"][st] / 60,
                         "Peak GPU (GB)": t["stage_peak_gb"][st]})
        rows.append({"Run": f"{t['n_visual_tokens']} tokens", "Stage": "final test generation (4 conditions x 320 q)",
                     "Time (min)": sum(e["suites"][s][c]["gen_time_per_answer_s"] * e["suites"][s]["n_questions"] for s in e["suites"] for c in CONDS) / 60,
                     "Peak GPU (GB)": e["gpu"]["peak_alloc_gb"]})
    rows.append({"Run": "shared", "Stage": f"feature caching ({fc['n_images']} images; fetch {fc['image_fetch_s'] / 60:.1f} min)", "Time (min)": fc["encode_s"] / 60,
                 "Peak GPU (GB)": fc["gpu"]["peak_alloc_gb"]})
    cols = ["Run", "Stage", "Trainable (M)", "Connector (M)", "LoRA (M)", "Time (min)", "Peak GPU (GB)"]
    f2 = lambda v: f"{v:.2f}"
    write_table(rows, cols, "t2_resources", {c: f2 for c in cols[2:]})
    comp = []
    for run in ["main", "pool49"]:
        t, e = jd(f"{R}/{run}/train_summary.json"), jd(f"{R}/{run}/final_eval.json")
        r = {"Visual tokens": t["n_visual_tokens"], "Train time (h)": t["train_time_s"] / 3600, "Peak GPU (GB)": max(t["stage_peak_gb"].values()),
             "Skipped updates": len(t["skipped"]), "Calibration scale": jd(f"{R}/{run}/calibration.json")["scale"]}
        for s in ["grounding", "vqa"]:
            S = e["suites"][s]
            r[f"{s} acc"] = 100 * S["correct"]["accuracy"]
            r[f"{s} adv (pp)"] = S["advantage_correct_vs_complementary"]["accuracy_pp"]
            r[f"{s} pair acc"] = 100 * S["correct"]["pair_accuracy"]
            r[f"{s} ms/answer"] = 1000 * S["correct"]["gen_time_per_answer_s"]
        comp.append(r)
    cols = list(comp[0])
    write_table(comp, cols, "t2_tokens_196_vs_49", {c: (lambda v: f"{v:.2f}") if isinstance(comp[0][c], float) else str for c in cols})
    return rows, comp


def qualitative(run="main", k=3):
    """selection rule (fixed, no cherry-picking): within each suite, test pairs in file order; take the first k pairs
    in each category: (A) both sides right with the correct image and both swapped predictions follow the partner;
    (B) the prediction is identical under both images (language-prior behaviour); (C) correct-image error."""
    rows = jl(f"{R}/{run}/final_test_predictions.jsonl")
    by = {}
    for r in rows:
        by.setdefault((r["suite"], r["pair_id"]), {}).setdefault(r["cond"], {})[r["side"]] = r
    out = {"rule": qualitative.__doc__, "examples": {}}
    for suite in ["grounding", "vqa"]:
        cats = {"A_follows_image": [], "B_ignores_image": [], "C_correct_image_error": []}
        for (s, pid), d in by.items():
            if s != suite:
                continue
            c, w = d["correct"], d["complementary"]
            both_right = all(c[i]["score"] >= 0.999 for i in (0, 1))
            if both_right and all(w[i]["follows_partner"] for i in (0, 1)):
                cats["A_follows_image"].append(pid)
            elif c[0]["prediction_norm"] == c[1]["prediction_norm"]:
                cats["B_ignores_image"].append(pid)
            elif any(c[i]["score"] < 0.5 for i in (0, 1)):
                cats["C_correct_image_error"].append(pid)
        ex = {}
        for cat, pids in cats.items():
            ex[cat] = {"count": len(pids), "pairs": []}
            for pid in pids[:k]:
                d = by[(suite, pid)]
                ex[cat]["pairs"].append({"pair_id": pid, "question": d["correct"][0]["question"], "type": d["correct"][0]["type"],
                                         "sides": [{"image": d["correct"][i]["image"], "reference": d["correct"][i]["reference"],
                                                    "pred_correct_image": d["correct"][i]["prediction"], "pred_partner_image": d["complementary"][i]["prediction"],
                                                    "pred_mismatched": d["mismatched"][i]["prediction"], "pred_no_image": d["noimage"][i]["prediction"]} for i in (0, 1)]})
        out["examples"][suite] = ex
    json.dump(out, open(f"{TAB}/t2_qualitative_{run}.json", "w"), indent=1)
    return {s: {c: v["count"] for c, v in e.items()} for s, e in out["examples"].items()}


def main():
    dv = curves("main")
    curves("pool49")
    final_tables("main")
    final_tables("pool49")
    res, comp = resources()
    q = qualitative("main")
    print("dev @ stage ends:", [(d["update"], round(d["caption_loss"], 3), round(d["grounding_dev"]["swap_penalty"], 2), round(d["vqa_dev"]["swap_penalty"], 2)) for d in dv if d["update"] in (0, 2250, 3750, 4500)])
    print("196 vs 49:", json.dumps(comp, indent=0, default=str)[:900])
    print("qualitative category counts:", q)


if __name__ == "__main__":
    main()
