"""Task 3 report assets from results/task3/offline + live_summary.json.

python -m analysis.task3
Figures: accuracy/oracle vs candidates and vs token cost; adaptive vs fixed budgets.
Tables: method comparison (N=8, 32) with paired bootstrap, failure decomposition, signal analysis, adaptive results,
live-generation comparison. Qualitative cases use fixed selection rules (documented in the JSON).
"""
import json

import pandas as pd

from analysis.common import INK2, NEUTRAL, SLOTS, TAB, jd, plt, save, write_table

O = "results/task3/offline"
DSN = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
MN = {"first": "FirstAns", "sc": "Self-consistency", "max_logp": "Max mean log-prob", "max_selfcert": "Max self-certainty",
      "rwc_selfcert": "Self-certainty + RWC", "shortest": "Shortest (length control)", "max_vision": "Max vision", "rwc_vision": "Vision + RWC",
      "max_visconf": "Max VisConf", "rwc_visconf": "VisConf + RWC", "max_tokenfusion": "Max token-fusion", "rwc_tokenfusion": "Token-fusion + RWC"}
FIGM = [("sc", SLOTS[0], "-", "o"), ("first", NEUTRAL, ":", "x"), ("max_selfcert", SLOTS[1], "--", "s"), ("rwc_selfcert", SLOTS[2], "-.", "^"),
        ("rwc_visconf", SLOTS[4], (0, (5, 1, 1, 1)), "v")]
BUD = [1, 2, 4, 8, 16, 32]


def scaling(S):
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 4.2))
    for j, ds in enumerate(["math", "mathvista"]):
        M = S[ds]["methods"]["eval"]
        ax, ax2 = axs[0, j], axs[1, j]
        orc = [100 * M["sc"][str(n)]["oracle_coverage"] for n in BUD]
        ax.plot(BUD, orc, color=INK2, lw=1.1, ls=(0, (1, 1)), label="Oracle coverage")
        for m, col, ls, mk in FIGM:
            if m not in M:
                continue
            acc = [100 * M[m][str(n)]["accuracy"] for n in BUD]
            cost = [M[m][str(n)]["mean_token_cost"] / 1000 for n in BUD]
            ax.plot(BUD, acc, color=col, ls=ls, marker=mk, ms=3.5, label=MN[m])
            ax2.plot(cost, acc, color=col, ls=ls, marker=mk, ms=3.5, label=MN[m])
        ax2.plot([M["sc"][str(n)]["mean_token_cost"] / 1000 for n in BUD], orc, color=INK2, lw=1.1, ls=(0, (1, 1)))
        for a in (ax,):
            a.set_xscale("log", base=2)
            a.set_xticks(BUD, [str(b) for b in BUD])
            a.set_xlabel("candidates N")
        ax2.set_xscale("log")
        ax2.set_xlabel("generated tokens per question (thousands, log)")
        ax.set_title(f"({'ab'[j]}) {DSN[ds]}: accuracy vs N")
        ax2.set_title(f"({'cd'[j]}) {DSN[ds]}: accuracy vs token cost")
        ax.set_ylabel("held-out accuracy / coverage (%)")
        ax2.set_ylabel("held-out accuracy (%)")
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=6.3)
    save(fig, "t3_scaling")


def method_table(S):
    rows = []
    for ds in ["math", "mathvista"]:
        M, B = S[ds]["methods"]["eval"], S[ds]["bootstrap"]
        for m in [x for x in MN if x in M]:
            r = {"Dataset": DSN[ds].split(" ")[0], "Method": MN[m]}
            for n in [1, 4, 8, 32]:
                r[f"Acc@{n}"] = 100 * M[m][str(n)]["accuracy"]
            r["Tokens@32 (k)"] = M[m]["32"]["mean_token_cost"] / 1000
            d = B["N32"].get(f"{m}-sc")
            r["vs SC@32 (pp)"] = "{:+.1f} [{:+.1f}, {:+.1f}]".format(d["diff_pp"], *d["ci95_pp"]) if d else ("--" if m != "sc" else "ref")
            rows.append(r)
        rows.append({"Dataset": DSN[ds].split(" ")[0], "Method": "Oracle coverage", **{f"Acc@{n}": 100 * M["sc"][str(n)]["oracle_coverage"] for n in [1, 4, 8, 32]},
                     "Tokens@32 (k)": M["sc"]["32"]["mean_token_cost"] / 1000, "vs SC@32 (pp)": "--"})
        rows.append({"Dataset": DSN[ds].split(" ")[0], "Method": "Distinct answer groups", **{f"Acc@{n}": M["sc"][str(n)]["distinct_groups"] for n in [1, 4, 8, 32]},
                     "Tokens@32 (k)": None, "vs SC@32 (pp)": "--"})
    cols = ["Dataset", "Method", "Acc@1", "Acc@4", "Acc@8", "Acc@32", "Tokens@32 (k)", "vs SC@32 (pp)"]
    f1 = lambda v: f"{v:.1f}"
    write_table(rows, cols, "t3_methods", {c: f1 for c in cols[2:7]})
    boot = {ds: S[ds]["bootstrap"] for ds in S}
    json.dump(boot, open(f"{TAB}/t3_bootstrap.json", "w"), indent=1)


def decomposition(S):
    rows = []
    for ds in ["math", "mathvista"]:
        for m in ["sc", "rwc_selfcert"] + (["rwc_visconf"] if ds == "mathvista" else []):
            D = S[ds]["decomposition"]["eval"][m]
            for n in BUD:
                d = D[str(n)]
                rows.append({"Dataset": DSN[ds].split(" ")[0], "Selector": MN[m], "N": n, "Accuracy": 100 * d["accuracy"],
                             "No correct candidate (sampling)": 100 * d["no_correct_candidate"], "Correct available, missed (selection)": 100 * d["correct_available_but_missed"]})
    cols = list(rows[0])
    write_table(rows, cols, "t3_decomposition", {c: (lambda v: f"{v:.1f}") for c in cols[3:]})


def signals(G):
    rows = []
    for ds in ["math", "mathvista"]:
        g = G[ds]
        for k in g["quartile_gap"]:
            rows.append({"Dataset": DSN[ds].split(" ")[0], "Signal": k, "Top-bottom quartile gap, all q (pp)": 100 * g["quartile_gap"][k]["all_questions"],
                         "Gap, mixed q (pp)": 100 * g["quartile_gap"][k]["mixed_questions"], "Spearman w/ length": g["spearman_with_length"][k],
                         "Spearman w/ correct (mixed q)": g["spearman_with_correct"][k]})
    cols = list(rows[0])
    write_table(rows, cols, "t3_signals", {c: (lambda v: f"{v:+.1f}") if "pp" in c else (lambda v: f"{v:+.3f}") for c in cols[2:]})
    extra = {ds: {k: v for k, v in G[ds].items() if k not in ("quartile_gap", "spearman_with_length", "spearman_with_correct")} for ds in G}
    json.dump(extra, open(f"{TAB}/t3_signals_extra.json", "w"), indent=1)


def adaptive_assets(A, S):
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for j, ds in enumerate(["math", "mathvista"]):
        a = A[ds]
        fx = a["heldout_fixed_sc"]
        for n in BUD:
            rows.append({"Dataset": DSN[ds].split(" ")[0], "Policy": f"fixed SC, N={n}", "tau": None, "gamma": None, "Accuracy": 100 * fx[str(n)]["accuracy"],
                         "Tokens (k)": fx[str(n)]["mean_token_cost"] / 1000, "Rollouts": n, "Stops at 4/8/16/32 (%)": "--"})
        ax = axs[j]
        ax.plot([fx[str(n)]["mean_token_cost"] / 1000 for n in BUD], [100 * fx[str(n)]["accuracy"] for n in BUD], color=SLOTS[0], marker="o", ms=3.5, label="fixed-N self-consistency")
        for n in BUD:
            ax.annotate(f"N={n}", (fx[str(n)]["mean_token_cost"] / 1000, 100 * fx[str(n)]["accuracy"]), fontsize=5.5, color=INK2, xytext=(3, 3), textcoords="offset points")
        for fam, col, mk in [("agreement", SLOTS[1], "s"), ("quality_aware", SLOTS[4], "D")]:
            f = a["families"][fam]
            h, sel = f["heldout"], f["selected"]
            sd = h["stop_distribution"]
            rows.append({"Dataset": DSN[ds].split(" ")[0], "Policy": fam.replace("_", "-"), "tau": sel["tau"], "gamma": sel["gamma"], "Accuracy": 100 * h["accuracy"],
                         "Tokens (k)": h["mean_token_cost"] / 1000, "Rollouts": h["mean_rollouts"],
                         "Stops at 4/8/16/32 (%)": "/".join(f"{100 * sd.get(str(k), sd.get(k, 0)):.0f}" for k in [4, 8, 16, 32])})
            ax.plot(h["mean_token_cost"] / 1000, 100 * h["accuracy"], marker=mk, color=col, ms=6, ls="none", label=f"{fam.replace('_', '-')} stopping")
        ax.set_xscale("log")
        ax.set_xlabel("generated tokens per question (thousands, log)")
        ax.set_ylabel("held-out accuracy (%)")
        ax.set_title(f"({'ab'[j]}) {DSN[ds]}")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=6.5)
    save(fig, "t3_adaptive")
    cols = ["Dataset", "Policy", "tau", "gamma", "Accuracy", "Tokens (k)", "Rollouts", "Stops at 4/8/16/32 (%)"]
    write_table(rows, cols, "t3_adaptive", {"Accuracy": lambda v: f"{v:.1f}", "Tokens (k)": lambda v: f"{v:.2f}", "Rollouts": lambda v: f"{v:.1f}"})
    extra = {ds: {fam: {k: v for k, v in A[ds]["families"][fam]["heldout"].items()} | {"selected": A[ds]["families"][fam]["selected"],
                                                                                         "dev_fixed32_sc": A[ds]["dev_fixed32_sc"]}
                  for fam in A[ds]["families"]} for ds in A}
    json.dump(extra, open(f"{TAB}/t3_adaptive_extra.json", "w"), indent=1, default=str)


def live():
    L = jd("results/task3/live_summary.json")
    rows = []
    for k, name in [("single", "Single response (3,072 cap)"), ("parallel", "Parallel x3 (1,024 each) + SC"), ("refinement", "Initial (1,024) + revision (2,048)")]:
        s = L["strategies"][k]
        rows.append({"Strategy": name, "Accuracy": 100 * s["accuracy"], "Input tok": s["mean_input_tokens"], "Output tok": s["mean_output_tokens"],
                     "Time (s)": s["mean_time_s"], "Call trunc. %": 100 * s["call_truncation_rate"], "Call extract. %": 100 * s["call_extraction_rate"],
                     "Final answer extracted %": 100 * s["final_answer_extracted_rate"]})
    cols = list(rows[0])
    write_table(rows, cols, "t3_live", {c: (lambda v: f"{v:.1f}") if c not in ("Input tok", "Output tok") else (lambda v: f"{v:.0f}") for c in cols[1:]})
    return L


def cases(S):
    """fixed rules (eval split, permutation order, order 0):
    (1) selection disagreement at N=32: first 3 questions where SC and self-certainty+RWC (VisConf+RWC on MathVista) pick
        different groups and exactly one is correct;
    (2) premature stopping: first 3 held-out questions where the selected quality-aware policy stopped before 32 with a wrong
        answer although fixed-32 SC was right;
    (3) refinement: every live question whose answer changed correctness after revision."""
    sel = pd.read_parquet(f"{O}/selections.parquet")
    splits = jd("data/task3/splits.json")
    prep = jd(f"{O}/prepared.json")
    out = {"rules": cases.__doc__, "selection": {}, "premature": {}, "refinement": []}
    for ds in ["math", "mathvista"]:
        other = "rwc_visconf" if ds == "mathvista" else "rwc_selfcert"
        s = sel[(sel.dataset == ds) & (sel.N == 32) & (sel.order == 0) & sel.method.isin(["sc", other])].set_index(["sample_id", "method"])
        found = []
        for sid in splits[ds]["eval"]:
            a, b = s.loc[(sid, "sc")], s.loc[(sid, other)]
            if a.answer_group != b.answer_group and a.correct != b.correct:
                g = prep[ds]["groups"][sid]
                labels = g["labels"]
                found.append({"sample_id": sid, "sc_answer": g["reps"].get(a.answer_group, a.answer_group), "sc_correct": bool(a.correct),
                              f"{other}_answer": g["reps"].get(b.answer_group, b.answer_group), f"{other}_correct": bool(b.correct),
                              "votes": {g["reps"].get(l, l): labels.count(l) for l in dict.fromkeys(labels)},
                              "group_correct": {g["reps"].get(l, l): v for l, v in g["group_correct"].items()}})
            if len(found) == 3:
                break
        out["selection"][ds] = found
        ad = pd.read_parquet(f"{O}/adaptive_{ds}_quality_aware_heldout.parquet")
        pm = ad[(ad.order == 0) & (ad.stop_n < 32) & (~ad.correct) & ad.sc32_correct].head(3)
        out["premature"][ds] = pm.to_dict("records")
    L = jd("results/task3/live_summary.json")
    for e in L["per_question"]:
        if e["correct"]["initial"] != e["correct"]["refinement"]:
            out["refinement"].append({"sample_id": e["sample_id"], "gold": e["gold"], "initial": e["answers"]["initial"], "revised": e["answers"]["revised"],
                                      "initial_correct": e["correct"]["initial"], "revised_correct": e["correct"]["refinement"]})
    json.dump(out, open(f"{TAB}/t3_cases.json", "w"), indent=1, default=str)
    return out


def main():
    S, G, A = jd(f"{O}/summary.json"), jd(f"{O}/signals.json"), jd(f"{O}/adaptive.json")
    scaling(S)
    method_table(S)
    decomposition(S)
    signals(G)
    adaptive_assets(A, S)
    live()
    c = cases(S)
    for ds in S:
        M = S[ds]["methods"]["eval"]
        print(ds, {m: round(100 * M[m]["32"]["accuracy"], 1) for m in M}, "oracle", round(100 * M["sc"]["32"]["oracle_coverage"], 1))
        print("  adaptive:", {f: (A[ds]["families"][f]["selected"]["tau"], A[ds]["families"][f]["selected"]["gamma"], round(100 * A[ds]["families"][f]["heldout"]["accuracy"], 1),
                                 round(A[ds]["families"][f]["heldout"]["mean_rollouts"], 1)) for f in A[ds]["families"]})
    print("cases:", {k: (len(v) if isinstance(v, list) else {d: len(x) for d, x in v.items()}) for k, v in c.items() if k != "rules"})


if __name__ == "__main__":
    main()
