"""Task 1 report assets: benchmark table (pp changes), loss + regularisation figure, resources, diagnostics,
qualitative grouping and the 42-response annotation sheet.

python -m analysis.task1 [--exports kaggle_output/task1/pa3/outputs/task1]
"""
import argparse
import csv
import json
import os
import re

import numpy as np

from analysis.common import INK2, NEUTRAL, T1, T1_COND, TAB, jd, jl, plt, save, write_table

R = "results/task1"
BENCH = [("gsm8k_flexible_em", None, "GSM8K"), ("hellaswag_acc_norm", "hellaswag_acc", "HellaSwag"),
         ("boolq_acc", None, "BoolQ"), ("arc_easy_acc_norm", "arc_easy_acc", "ARC-E")]
RUNS = ["fullft", "l2sp", "lora4", "lora16", "oplora"]
CONDS = list(T1_COND)


def benchmarks():
    ev = {c: jd(f"{R}/eval_{c}.json") for c in CONDS}
    base = ev["baseline"]["primary"]
    rows = []
    for c in CONDS:
        p, ex = ev[c]["primary"], ev[c]["lm_eval"]["gsm8k_extra"]
        r = {"Condition": T1_COND[c]}
        for k, raw, name in BENCH:
            r[name] = 100 * p[k]
            r[f"d{name}"] = 100 * (p[k] - base[k])
            if raw:
                r[f"{name} raw"] = 100 * p[raw]
        r["GSM8K strict"] = 100 * p["gsm8k_strict_em"]
        r["GSM8K trunc %"] = 100 * ex["truncation_rate"]
        r["GSM8K unextract. %"] = 100 * ex["flexible_unextractable_rate"]
        r["GSM8K mean tokens"] = ex["mean_resp_tokens"]
        r["mean non-math d"] = np.mean([r[f"d{n}"] for _, _, n in BENCH[1:]])
        rows.append(r)
    cols = ["Condition"] + [x for _, _, n in BENCH for x in (n, f"d{n}")] + ["mean non-math d", "HellaSwag raw", "ARC-E raw", "GSM8K trunc %", "GSM8K unextract. %", "GSM8K mean tokens"]
    f1 = lambda v: f"{v:.1f}"
    sg = lambda v: f"{v:+.1f}"
    write_table(rows, cols, "t1_benchmarks", {c: (sg if c.startswith("d") or c == "mean non-math d" else f1) for c in cols if c != "Condition"})
    return rows, ev


def gsm_breakdown(ev):
    """separate answer correctness from extractability/truncation changes vs baseline (per example)"""
    base = {r["doc_id"]: r for r in jl(f"{R}/samples/baseline_gsm8k.jsonl")}
    out = {}
    for c in CONDS[1:]:
        rows = jl(f"{R}/samples/{c}_gsm8k.jsonl")
        b2c = sum(1 for r in rows if not base[r["doc_id"]]["exact_match_flexible-extract"] and r["exact_match_flexible-extract"])
        c2b = sum(1 for r in rows if base[r["doc_id"]]["exact_match_flexible-extract"] and not r["exact_match_flexible-extract"])
        out[c] = {"wrong_to_right": b2c, "right_to_wrong": c2b, "net": b2c - c2b,
                  "wrong_answers_truncated": sum(1 for r in rows if not r["exact_match_flexible-extract"] and r.get("truncated")),
                  "boxed_or_answer_is_style": float(np.mean([("The answer is" in r["response"]) or ("\\boxed" in r["response"]) for r in rows])),
                  "mean_tokens": float(np.mean([r["n_resp_tokens"] for r in rows]))}
    base_rows = list(base.values())
    out["baseline"] = {"boxed_or_answer_is_style": float(np.mean([("The answer is" in r["response"]) or ("\\boxed" in r["response"]) for r in base_rows])),
                       "mean_tokens": float(np.mean([r["n_resp_tokens"] for r in base_rows]))}
    json.dump(out, open(f"{TAB}/t1_gsm8k_transitions.json", "w"), indent=1)
    return out


def ema(x, a=0.05):
    y, m = [], x[0]
    for v in x:
        m = a * v + (1 - a) * m
        y.append(m)
    return np.array(y)


def loss_figure():
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    for r in RUNS:
        log = jl(f"{R}/train_log_{r}.jsonl")
        u = np.array([x["update"] for x in log])
        l = np.array([x["sft_loss"] for x in log])
        name, col, ls, mk = T1[r]
        ax.plot(u, ema(l), color=col, ls=ls, lw=1.3, label=name)
        ax.plot(u, l, color=col, lw=0.4, alpha=0.18)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss (response tokens, EMA 0.05)")
    ax.set_ylim(0.12, 0.42)
    ax.axvline(625, color=NEUTRAL, lw=0.7, ls=":")
    ax.text(630, 0.405, "midpoint", fontsize=6.5, color=INK2)
    ax.legend(ncol=2, loc="lower left")
    ax.set_title("(a) SFT loss, all five runs")
    ax = axs[1]
    log = jl(f"{R}/train_log_l2sp.jsonl")
    u = np.array([x["update"] for x in log])
    ax.plot(u, [x["lam_omega"] for x in log], color=T1["l2sp"][1], ls="--", label=r"$\lambda\Omega$ (L2-SP)")
    ax.plot(u, [x["total_loss"] - x["sft_loss"] for x in log], alpha=0)  # same series; keeps limits honest
    ax.set_xlabel("optimizer update")
    ax.set_ylabel(r"$\lambda\Omega$")
    ax.set_title(r"(b) L2-SP penalty $\lambda\Omega$, $\lambda=10^{-3}$")
    save(fig, "t1_loss")


def resources(exports_dir):
    rows, reg = [], {}
    for r in RUNS:
        t = jd(f"{R}/train_{r}.json")
        ov = t["overhead"]
        ex = {}
        for tag in ["mid", "final"]:
            p = f"{exports_dir}/{r}/{tag}/export_info.json"
            if os.path.exists(p):
                ex[tag] = jd(p)
        rc = ov.get("recovery_ckpt_bytes", {})
        row = {"Method": T1[r][0], "Trainable params (M)": t["trainable_params"] / 1e6, "Train time (h)": t["train_time_h"],
               "s/update": t["train_time_s"] / t["updates"], "Peak GPU mem (GB)": t["peak_alloc_gb"],
               "Recovery ckpt (GB)": (max(rc.values()) / 2**30) if rc else None,
               "Export (MB)": ex["final"]["export_bytes"] / 2**20 if "final" in ex else None, "Skipped updates": len(t["skipped_updates"])}
        if r == "l2sp":
            row["Method overhead"] = f"theta0 {ov['l2sp_ref_bytes'] / 2**30:.2f} GB bf16, copy {ov['l2sp_ref_transfer_s']:.2f} s, penalty {ov['l2sp_penalty_time_s'] / 60:.1f} min"
        elif r == "oplora":
            row["Method overhead"] = (f"bases {ov['oplora_basis_bytes'] / 2**20:.0f} MB, SVD {ov['oplora_svd_time_s']:.0f} s"
                                      + (f", export {ex['final']['export_time_s']:.2f} s" if "final" in ex else ""))
        else:
            row["Method overhead"] = "--"
        rows.append(row)
        if r in ("fullft", "l2sp"):
            log = jl(f"{R}/train_log_{r}.jsonl")
            reg[r] = {"displacement_mid": ex.get("mid", {}).get("displacement_l2"), "displacement_final": ex.get("final", {}).get("displacement_l2"),
                      "final_sft_loss_last50": float(np.mean([x["sft_loss"] for x in log[-50:]]))}
            if r == "l2sp":
                reg[r].update(omega_final=log[-1]["omega"], lam_omega_final=log[-1]["lam_omega"],
                              lam_omega_max=max(x["lam_omega"] for x in log), lam_omega_over_sft_final=log[-1]["lam_omega"] / log[-1]["sft_loss"])
            if reg[r]["displacement_final"] is not None:
                reg[r]["omega_implied_final"] = 0.5 * reg[r]["displacement_final"] ** 2
        if r in ("lora4", "lora16", "oplora"):
            reg.setdefault("adapters", {})[r] = {"init_A_checksum": ov.get("init_A_checksum"),
                                                "oplora_constraint": {k: v for k, v in ex.get("final", {}).items() if k.startswith("constraint")}}
    cols = ["Method", "Trainable params (M)", "Train time (h)", "s/update", "Peak GPU mem (GB)", "Recovery ckpt (GB)", "Export (MB)", "Skipped updates", "Method overhead"]
    write_table(rows, cols, "t1_resources", {"Trainable params (M)": lambda v: f"{v:.1f}", "Train time (h)": lambda v: f"{v:.2f}", "s/update": lambda v: f"{v:.2f}",
                                              "Peak GPU mem (GB)": lambda v: f"{v:.1f}", "Recovery ckpt (GB)": lambda v: f"{v:.2f}", "Export (MB)": lambda v: f"{v:.0f}"})
    json.dump(reg, open(f"{TAB}/t1_regularization.json", "w"), indent=1)
    return rows, reg


def diagnostics():
    dg = jd(f"{R}/diagnostics.json")
    order = ["fullft_mid", "fullft", "l2sp", "lora4", "lora16", "oplora"]
    rows = []
    for c in order:
        d = dg["checkpoints"][c]
        r = {"Checkpoint": T1_COND[c]}
        for li in ["0", "13", "27"]:
            r[f"rho L{li}"] = d[li]["rho_k"]
            r[f"eta L{li}"] = d[li]["eta"]
            r[f"Rm L{li}"] = d[li]["R_math_mean"]
            r[f"Rn L{li}"] = d[li]["R_nonmath_mean"]
            r[f"Rm/Rn L{li}"] = d[li]["R_math_over_nonmath"]
        rows.append(r)
    cols = ["Checkpoint"] + [f"{m} L{li}" for li in ["0", "13", "27"] for m in ["rho", "eta", "Rm", "Rn", "Rm/Rn"]]
    sci = lambda v: f"{v:.2e}"
    write_table(rows, cols, "t1_diagnostics", {c: (sci if (c.startswith(("eta", "Rm L", "Rn")) or c.startswith("rho")) else (lambda v: f"{v:.2f}")) for c in cols[1:]})
    # figure: rho_k, eta (log) and R math vs non-math per layer
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.1))
    x = np.arange(3)
    w = 0.13
    keys = {"fullft_mid": ("Full FT (mid)", "#9cc0ec"), "fullft": T1["fullft"][:2], "l2sp": T1["l2sp"][:2], "lora4": T1["lora4"][:2],
            "lora16": T1["lora16"][:2], "oplora": T1["oplora"][:2]}
    for j, c in enumerate(order):
        name, col = keys[c]
        d = dg["checkpoints"][c]
        axs[0].bar(x + (j - 2.5) * w, [max(d[li]["rho_k"], 1e-9) for li in ["0", "13", "27"]], w * 0.9, color=col, label=name)
        axs[1].bar(x + (j - 2.5) * w, [d[li]["eta"] for li in ["0", "13", "27"]], w * 0.9, color=col)
        axs[2].bar(x + (j - 2.5) * w, [d[li]["R_math_over_nonmath"] for li in ["0", "13", "27"]], w * 0.9, color=col)
    for ax, t in zip(axs, [r"(a) subspace interference $\rho_{128}$", r"(b) relative update $\eta$", r"(c) $\bar R_{math}/\bar R_{non\text{-}math}$"]):
        ax.set_xticks(x, ["L0", "L13", "L27"])
        ax.set_title(t)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[1].set_yticks([3e-3, 1e-2, 3e-2], ["0.003", "0.01", "0.03"])
    axs[1].minorticks_off()
    axs[2].axhline(1, color=NEUTRAL, lw=0.8, ls=":")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=6.5)
    save(fig, "t1_diagnostics")
    return rows


def qualitative():
    """168 responses grouped by question and format; annotation sheet for the first and last plain questions (42 rows)"""
    allq = {c: jd(f"{R}/qualitative/{c}.json")["responses"] for c in CONDS}
    grouped = {}
    for c, rs in allq.items():
        for r in rs:
            grouped.setdefault(r["question"], {}).setdefault(r["format"], {})[c] = {"response": r["response"], "n_tokens": r["n_tokens"], "truncated": r["truncated"]}
    json.dump(grouped, open(f"{R}/qualitative_grouped.json", "w"), indent=1)
    plain_order = [r["question"] for r in allq["baseline"] if r["format"] == "plain"]
    pick = [plain_order[0], plain_order[-1]]
    math_terms = re.compile(r"\\boxed|\\frac|\$[^$]+\$|=\s*\d|\bequation\b|\bcalculate\b|\bstep \d\b|\bThe answer is\b", re.I)
    rows = []
    for q in pick:
        for fmt in ["plain", "json", "template"]:
            for c in CONDS:
                r = grouped[q][fmt][c]
                txt = r["response"]
                words = txt.split()
                rep = 1 - len(set(map(tuple, zip(words, words[1:], words[2:])))) / max(1, len(words) - 2)
                fmt_ok = ""
                if fmt == "json":
                    s = txt.strip().strip("`")
                    s = s[4:] if s.startswith("json") else s
                    try:
                        json.loads(s)
                        fmt_ok = "valid JSON"
                    except Exception:
                        fmt_ok = "invalid JSON"
                elif fmt == "template":
                    fmt_ok = "has 'x = ' fields" if re.search(r"^\w+\s*=", txt, re.M) else "no template fields"
                rows.append({"question": q, "format": fmt, "condition": T1_COND[c], "n_tokens": r["n_tokens"], "truncated": r["truncated"],
                             "auto_format_check": fmt_ok, "auto_trigram_repetition": round(rep, 3), "auto_math_markers": len(math_terms.findall(txt)),
                             "format_compliance (0-2)": "", "coherence_relevance (0-2)": "", "repetition (0-2)": "", "unneeded_math_framing (y/n)": "",
                             "inappropriate_content (y/n)": "", "excerpt / note": "", "response": txt})
    with open(f"{R}/qualitative_annotation_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return pick, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="kaggle_output/task1/pa3/outputs/task1")
    a = ap.parse_args()
    rows, ev = benchmarks()
    tr = gsm_breakdown(ev)
    loss_figure()
    res, reg = resources(a.exports)
    diagnostics()
    pick, n = qualitative()
    print("benchmarks:")
    for r in rows:
        print(f"  {r['Condition']:16s} GSM {r['GSM8K']:.1f} ({r['dGSM8K']:+.1f})  HS {r['HellaSwag']:.1f} ({r['dHellaSwag']:+.1f})  "
              f"BQ {r['BoolQ']:.1f} ({r['dBoolQ']:+.1f})  ARC {r['ARC-E']:.1f} ({r['dARC-E']:+.1f})  trunc {r['GSM8K trunc %']:.1f}%")
    print("regularization:", json.dumps(reg, indent=1, default=str)[:1500])
    print("resources:", [(r["Method"], round(r["Train time (h)"], 2), round(r["Peak GPU mem (GB)"], 1)) for r in res])
    print("gsm transitions:", {k: (v.get("net"), round(v["mean_tokens"])) for k, v in tr.items()})
    print("annotation sheet:", pick, n, "rows")


if __name__ == "__main__":
    main()
