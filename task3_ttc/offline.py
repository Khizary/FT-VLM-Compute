"""Task 3 offline analysis on the supplied rollouts (Steps 1, 2, 3, 5). CPU only.

python -m task3_ttc.offline --data localonly/pa3_data --meta localonly/pa3_data/mathvista_meta/testmini.json
Writes results/task3/offline/: prepared.json (groups, labels, alignment checks), orders.json, selections.parquet
(every question x order x budget x method), summary.json (accuracy/coverage/cost tables, failure decomposition,
bootstrap CIs), signals.json (Step 3), adaptive.json (Step 5).
Gold answers / correctness / future candidates are only used for grading and retrospective analysis.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from task3_ttc.answers import INVALID, NO_ANSWER, SCORER, group_math, group_mathvista

SEED = 6304
BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
N_BOOT = 1000
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]
CHECKS = [4, 8, 16]
OUT = "results/task3/offline"
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


# ---------------------------------------------------------------- loading
def load(data, ds, meta_path):
    f = f"{data}/{FOLDERS[ds]}"
    key = ["dataset", "sample_id", "rollout_id"]
    g = pd.read_parquet(f"{f}/generations.parquet")
    p = pd.read_parquet(f"{f}/probability_metrics.parquet")
    df = g.merge(p, on=key, how="inner", validate="one_to_one")
    if ds == "mathvista":
        a = pd.read_parquet(f"{f}/attention_metrics.parquet").rename(columns={"n_generated_text_tokens": "attn_prev_counts"})
        df = df.merge(a, on=key, how="inner", validate="one_to_one")
    assert len(df) == 32000, len(df)
    df = df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
    checks = {"rows": len(df), "questions": int(df.sample_id.nunique()), "rollouts_per_q": sorted(df.groupby("sample_id").size().unique().tolist())}
    T = df["n_generated_text_tokens"].to_numpy()
    lens = {c: df[c].map(len).to_numpy() for c in ["generated_token_ids", "logp", "kl_u_p"] + (["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total", "attn_prev_counts"] if ds == "mathvista" else [])}
    checks["alignment"] = {c: int((v != T).sum()) for c, v in lens.items()}
    checks["zero_length"] = int((T == 0).sum())
    if ds == "mathvista":
        checks["prev_counts_ok"] = int(sum(np.array_equal(np.asarray(x), np.arange(len(x))) for x in df["attn_prev_counts"]))
    assert all(v == 0 for v in checks["alignment"].values()), checks
    meta = json.load(open(meta_path)) if ds == "mathvista" else None
    return df, checks, meta


def _clean(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _group_math_job(args):
    sid, answers, gold = args
    return sid, group_math(answers, gold)


def math_groups(df):
    """math-verify grouping is the slow part: parallel over questions, cached on disk"""
    cache = f"{OUT}/groups_math_cache.json"
    if os.path.exists(cache):
        return {k: tuple(v) for k, v in json.load(open(cache)).items()}
    from multiprocessing import Pool

    jobs = [(sid, [_clean(x) for x in d.sort_values("rollout_id")["extracted_answer"]], d["gold_answer"].iloc[0]) for sid, d in df.groupby("sample_id", sort=True)]
    res = {}
    with Pool(os.cpu_count()) as pool:
        for k, (sid, g) in enumerate(pool.imap_unordered(_group_math_job, jobs, chunksize=4)):
            res[str(sid)] = g
            if k % 100 == 0:
                log(f"math grouping {k}/{len(jobs)}")
    json.dump(res, open(cache, "w"))
    return res


def prepare(df, ds, meta):
    """per question: labels, correctness, per-rollout scalars and token arrays"""
    Q = {}
    gold_mismatch = 0
    mg = math_groups(df) if ds == "math" else None
    for sid, d in df.groupby("sample_id", sort=True):
        d = d.sort_values("rollout_id")
        answers = [_clean(x) for x in d["extracted_answer"]]
        if ds == "math":
            labels, corr, reps = mg[str(sid)]
        else:
            m = meta[str(sid)]
            if str(m["answer"]) != str(d["gold_answer"].iloc[0]):
                gold_mismatch += 1
            labels, corr, reps = group_mathvista(answers, m)
        kl = [np.asarray(x, dtype=np.float64) for x in d["kl_u_p"]]
        lp = [np.asarray(x, dtype=np.float64) for x in d["logp"]]
        q = {"labels": labels, "correct": np.array([corr[l] for l in labels]), "group_correct": corr, "reps": reps,
             "T": d["n_generated_text_tokens"].to_numpy().astype(np.int64), "capped": d["hit_max_new_tokens"].to_numpy(),
             "L": np.array([x.mean() if len(x) else -np.inf for x in lp]), "K": np.array([x.mean() if len(x) else -np.inf for x in kl]),
             "kl": kl, "lp": lp, "answers": answers, "gold": d["gold_answer"].iloc[0]}
        if ds == "mathvista":
            v = []
            for a_i, a_p, a_g in zip(d["img_attn_total"], d["prompt_text_attn_total"], d["generated_text_attn_total"]):
                a_i, a_p, a_g = (np.asarray(x, dtype=np.float64) for x in (a_i, a_p, a_g))
                v.append(a_i / np.maximum(a_i + a_p + a_g, 1e-12))
            q["v"] = v
            q["V"] = np.array([x.mean() if len(x) else -np.inf for x in v])
            q["meta"] = {k: meta[str(sid)].get(k) for k in ["question_type", "answer_type", "precision"]}
        Q[str(sid)] = q
    return Q, gold_mismatch


# ---------------------------------------------------------------- selection rules
def avg_rank_desc(s):
    """average ranks, rank 1 = best (largest)"""
    s = np.asarray(s, dtype=np.float64)
    gt = (s[None, :] > s[:, None]).sum(1)
    eq = (s[None, :] == s[:, None]).sum(1)
    return 1 + gt + (eq - 1) / 2


def vote(labels, weights):
    """weighted vote; ties broken by earliest supporting rollout in the current order"""
    tot, first = {}, {}
    for i, (l, w) in enumerate(zip(labels, weights)):
        tot[l] = tot.get(l, 0.0) + w
        first.setdefault(l, i)
    best = max(tot.values())
    win = min((l for l in tot if np.isclose(tot[l], best, rtol=0, atol=1e-12)), key=lambda l: first[l])
    vals = sorted(tot.values(), reverse=True)
    return win, tot, vals


def rwc(labels, score):
    N = len(labels)
    if N == 1:
        return labels[0], 1.0
    r = avg_rank_desc(score)
    w = (N - r) / (N - 1)
    win, tot, vals = vote(labels, w)
    margin = (vals[0] - (vals[1] if len(vals) > 1 else 0.0)) / max(sum(vals), 1e-12)
    return win, margin


def calibrate(kl_list, idx):
    """VisConf calibration pooled over the current candidate set only (linear-interpolation percentiles)"""
    pool = np.concatenate([kl_list[i] for i in idx])
    q05, q95 = np.percentile(pool, [5, 95])
    den = max(q95 - q05, 1e-8)
    return [np.clip((kl_list[i] - q05) / den, 0, 1) for i in idx]


def prefix_stats(q, idx, vis):
    """everything the selectors and stopping rules need for one candidate prefix (rollout ids `idx`, in order)"""
    labels = [q["labels"][i] for i in idx]
    N = len(idx)
    out = {"labels": labels}
    sc_win, tot, vals = vote(labels, np.ones(N))
    out["sc"] = sc_win
    out["sc_frac"] = tot[sc_win] / N
    out["sc_unique"] = len(vals) == 1 or vals[0] > vals[1]
    sel = {"first": labels[0], "sc": sc_win}
    sel["max_logp"] = labels[int(np.argmax(q["L"][idx]))]
    sel["max_selfcert"] = labels[int(np.argmax(q["K"][idx]))]
    sel["shortest"] = labels[int(np.argmin(q["T"][idx]))]
    sel["rwc_selfcert"], out["margin_selfcert"] = rwc(labels, q["K"][idx])
    if vis:
        V = q["V"][idx]
        cs = calibrate(q["kl"], idx)
        C = np.array([c.mean() if len(c) else -np.inf for c in cs])
        Zt = np.array([(c * q["v"][i]).mean() if len(c) else -np.inf for c, i in zip(cs, idx)])
        Z = V * C
        sel["max_vision"] = labels[int(np.argmax(V))]
        sel["rwc_vision"], _ = rwc(labels, V)
        sel["max_visconf"] = labels[int(np.argmax(Z))]
        sel["rwc_visconf"], out["margin_visconf"] = rwc(labels, Z)
        sel["max_tokenfusion"] = labels[int(np.argmax(Zt))]
        sel["rwc_tokenfusion"], _ = rwc(labels, Zt)
    out["sel"] = sel
    return out


def method_cost(q, idx, m):
    return int(q["T"][idx[0]]) if m == "first" else int(q["T"][idx].sum())


# ---------------------------------------------------------------- main loops
def run_dataset(ds, Q, splits, orders):
    vis = ds == "mathvista"
    rows, stats = [], {}
    for k, (sid, q) in enumerate(sorted(Q.items())):
        split = "dev" if sid in splits["dev"] else "eval"
        for o, perm in enumerate(orders[sid]):
            for N in BUDGETS:
                idx = np.asarray(perm[:N])
                st = prefix_stats(q, idx, vis)
                stats[(sid, o, N)] = st
                cov = bool(q["correct"][idx].any())
                n_groups = len(set(st["labels"]))
                for m, lab in st["sel"].items():
                    rows.append((ds, sid, split, o, N, m, lab, bool(q["group_correct"][lab]), method_cost(q, idx, m), cov, n_groups,
                                 ",".join(map(str, idx.tolist()))))
        if k % 200 == 0:
            log(f"{ds}: {k}/1000 questions")
    sel = pd.DataFrame(rows, columns=["dataset", "sample_id", "split", "order", "N", "method", "answer_group", "correct", "token_cost", "oracle", "n_groups", "candidate_ids"])
    return sel, stats


def boot_diff(a, b, rng, n=N_BOOT):
    """paired question-level bootstrap of mean(a - b); a, b are per-question order-averaged accuracies"""
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return {"diff_pp": float(100 * d.mean()), "ci95_pp": [float(100 * np.percentile(m, 2.5)), float(100 * np.percentile(m, 97.5))]}


def summarize(sel, ds, rng):
    out = {"methods": {}, "decomposition": {}, "bootstrap": {}}
    for split in ["eval", "dev"]:
        s = sel[sel.split == split]
        perq = s.groupby(["method", "N", "sample_id"]).agg(acc=("correct", "mean"), cost=("token_cost", "mean"), oracle=("oracle", "mean"), groups=("n_groups", "mean")).reset_index()
        tab = perq.groupby(["method", "N"]).agg(acc=("acc", "mean"), cost=("cost", "mean"), oracle=("oracle", "mean"), groups=("groups", "mean")).reset_index()
        out["methods"][split] = {m: {int(r.N): {"accuracy": r.acc, "oracle_coverage": r.oracle, "mean_token_cost": r.cost, "distinct_groups": r.groups}
                                     for r in d.itertuples()} for m, d in tab.groupby("method")}
        # sampling vs selection failure (self-consistency and others), per (question, order) then averaged
        dec = {}
        for m, d in s.groupby("method"):
            d = d.assign(no_correct=~d.oracle, missed=d.oracle & ~d.correct)
            pq = d.groupby(["N", "sample_id"])[["no_correct", "missed", "correct"]].mean().groupby("N").mean()
            dec[m] = {int(n): {"no_correct_candidate": r.no_correct, "correct_available_but_missed": r.missed, "accuracy": r.correct} for n, r in pq.iterrows()}
        out["decomposition"][split] = dec
        if split == "eval":
            piv = perq[perq.N == 32].pivot(index="sample_id", columns="method", values="acc")
            pairs = [("sc", "first"), ("max_selfcert", "sc"), ("rwc_selfcert", "sc"), ("rwc_selfcert", "max_selfcert"), ("max_logp", "sc"), ("shortest", "sc")]
            if ds == "mathvista":
                pairs += [("rwc_visconf", "sc"), ("rwc_visconf", "max_visconf"), ("max_visconf", "max_selfcert"), ("rwc_visconf", "rwc_selfcert"),
                          ("rwc_vision", "sc"), ("rwc_tokenfusion", "rwc_visconf"), ("max_tokenfusion", "max_visconf")]
            out["bootstrap"]["N32"] = {f"{a}-{b}": boot_diff(piv[a].to_numpy(), piv[b].to_numpy(), rng) for a, b in pairs}
            for N in [4, 8]:
                piv = perq[perq.N == N].pivot(index="sample_id", columns="method", values="acc")
                out["bootstrap"][f"N{N}"] = {f"{a}-{b}": boot_diff(piv[a].to_numpy(), piv[b].to_numpy(), rng) for a, b in pairs}
    return out


# ---------------------------------------------------------------- Step 3
def quartile_gap(score, correct):
    """correctness in the top quartile minus bottom quartile of a score within one question (stable by rollout id)"""
    o = np.argsort(-score, kind="stable")
    k = len(score) // 4
    return correct[o[:k]].mean() - correct[o[-k:]].mean()


def spearman(a, b):
    ra, rb = avg_rank_desc(a), avg_rank_desc(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def signals(Q, ds, splits, n_examples=3):
    vis = ds == "mathvista"
    res = {"quartile_gap": {}, "spearman_with_length": {}, "spearman_with_correct": {}}
    scores = {"neg_length": lambda q, c: -q["T"].astype(float), "logp": lambda q, c: q["L"], "selfcert": lambda q, c: q["K"]}
    if vis:
        scores.update(vision=lambda q, c: q["V"], visconf=lambda q, c: q["V"] * c["C"], tokenfusion=lambda q, c: c["Zt"])
    per = {k: {"all": [], "mixed": []} for k in scores}
    sp_len = {k: [] for k in scores}
    sp_cor = {k: [] for k in scores}
    cv_rollout, cov_check = [], []
    for sid, q in sorted(Q.items()):
        if sid not in splits["eval"]:
            continue
        idx = np.arange(32)
        c = {}
        if vis:
            cs = calibrate(q["kl"], idx)
            c["C"] = np.array([x.mean() for x in cs])
            c["Zt"] = np.array([(x * v).mean() for x, v in zip(cs, q["v"])])
            for x, v in zip(cs, q["v"]):
                if len(x) > 2 and x.std() > 0 and v.std() > 0:
                    cv_rollout.append(float(np.corrcoef(x, v)[0, 1]))
            cov = np.array([np.mean(x * v) - x.mean() * v.mean() for x, v in zip(cs, q["v"])])
            cov_check.append(float(np.abs((c["Zt"] - q["V"] * c["C"]) - cov).max()))
        corr = q["correct"].astype(float)
        mixed = 0 < corr.mean() < 1
        for k, f in scores.items():
            s = f(q, c)
            gap = quartile_gap(s, corr)
            per[k]["all"].append(gap)
            if mixed:
                per[k]["mixed"].append(gap)
                sp_cor[k].append(spearman(s, corr))
            sp_len[k].append(spearman(s, q["T"].astype(float)))
    for k in scores:
        res["quartile_gap"][k] = {"all_questions": float(np.mean(per[k]["all"])), "mixed_questions": float(np.mean(per[k]["mixed"])), "n_mixed": len(per[k]["mixed"])}
        res["spearman_with_length"][k] = float(np.nanmean(sp_len[k]))
        res["spearman_with_correct"][k] = float(np.nanmean(sp_cor[k]))
    if vis:
        cv = np.array(cv_rollout)
        res["confidence_vision_corr_within_rollout"] = {"mean": float(cv.mean()), "median": float(np.median(cv)), "q25": float(np.percentile(cv, 25)),
                                                       "q75": float(np.percentile(cv, 75)), "frac_positive": float((cv > 0).mean()), "n_rollouts": len(cv)}
        res["identity_Ztoken_minus_Z_equals_cov_max_abs_err"] = float(max(cov_check))
    # fixed illustrative trajectories: first `n_examples` eval questions (permutation order) with a correct and an incorrect rollout;
    # for each, the highest- and lowest-self-certainty rollouts
    ex = []
    for sid in splits["eval"]:
        q = Q[sid]
        if 0 < q["correct"].mean() < 1:
            for tag, i in [("top_selfcert", int(np.argmax(q["K"]))), ("bottom_selfcert", int(np.argmin(q["K"])))]:
                e = {"sample_id": sid, "rollout_id": i, "tag": tag, "correct": bool(q["correct"][i]), "T": int(q["T"][i]), "answer": q["answers"][i], "gold": q["gold"],
                     "logp": q["lp"][i].round(4).tolist(), "kl": q["kl"][i].round(4).tolist()}
                if vis:
                    e["v"] = q["v"][i].round(5).tolist()
                ex.append(e)
            if len(ex) >= 2 * n_examples:
                break
    res["selection_rule_examples"] = f"first {n_examples} eval questions in permutation order with both correct and incorrect rollouts; highest/lowest self-certainty rollout of each"
    return res, ex


# ---------------------------------------------------------------- Step 5
def adaptive(Q, stats, ds, splits):
    vis = ds == "mathvista"
    mkey = "margin_visconf" if vis else "margin_selfcert"
    rkey = "rwc_visconf" if vis else "rwc_selfcert"

    def run_policy(sid, o, tau, gamma, quality):
        q = Q[sid]
        for n in CHECKS:
            st = stats[(sid, o, n)]
            ok = st["sc_unique"] and st["sc"] != NO_ANSWER and st["sc_frac"] >= tau - 1e-12
            if quality:
                ok = ok and st["sel"][rkey] == st["sc"] and st[mkey] >= gamma - 1e-12
            if ok:
                return n, st["sc"]
        return 32, stats[(sid, o, 32)]["sc"]

    def evaluate(qids, tau, gamma, quality, keep_rows=False):
        acc, cost, used, rows = [], [], [], []
        for sid in qids:
            q = Q[sid]
            a_q, c_q, u_q = [], [], []
            for o in range(N_ORDERS):
                if tau is None:  # always-32 fallback
                    n, lab = 32, stats[(sid, o, 32)]["sc"]
                else:
                    n, lab = run_policy(sid, o, tau, gamma, quality)
                perm = ORDERS[ds][sid][o]
                a_q.append(q["group_correct"][lab])
                c_q.append(int(q["T"][np.asarray(perm[:n])].sum()))
                u_q.append(n)
                if keep_rows:
                    rows.append({"sample_id": sid, "order": o, "stop_n": n, "answer_group": lab, "correct": bool(q["group_correct"][lab]),
                                 "token_cost": c_q[-1], "sc32_correct": bool(q["group_correct"][stats[(sid, o, 32)]["sc"]]),
                                 "pool_frac_correct": float(q["correct"].mean()), "sc32_support": stats[(sid, o, 32)]["sc_frac"]})
            acc.append(np.mean(a_q)); cost.append(np.mean(c_q)); used.append(np.mean(u_q))
        return float(np.mean(acc)), float(np.mean(cost)), float(np.mean(used)), rows

    dev, ev = splits["dev"], splits["eval"]
    base_acc, base_cost, _, _ = evaluate(dev, None, None, False)
    out = {"dev_fixed32_sc": {"accuracy": base_acc, "mean_token_cost": base_cost}, "families": {}}
    for fam, quality in [("agreement", False), ("quality_aware", True)]:
        grid = [(None, None)] + [(t, g) for t in TAUS for g in (GAMMAS if quality else [None])]
        cands = []
        for t, g in grid:
            a, c, u, _ = evaluate(dev, t, g if g is not None else 0.0, quality)
            cands.append({"tau": t, "gamma": g, "dev_accuracy": a, "dev_mean_token_cost": c, "dev_mean_rollouts": u})
        feas = [x for x in cands if x["dev_accuracy"] >= base_acc - 0.01 - 1e-12]
        best = sorted(feas, key=lambda x: (x["dev_mean_token_cost"], -x["dev_accuracy"], x["tau"] if x["tau"] is not None else 9, x["gamma"] if x["gamma"] is not None else 9))[0]
        a, c, u, rows = evaluate(ev, best["tau"], best["gamma"] if best["gamma"] is not None else 0.0, quality, keep_rows=True)
        r = pd.DataFrame(rows)
        early_wrong = r[(r.stop_n < 32) & (~r.correct)]
        out["families"][fam] = {
            "dev_grid": cands, "selected": best, "rule": "lowest dev cost with dev acc >= fixed-32 SC - 1pp; ties: higher acc, smaller tau, smaller gamma",
            "heldout": {"accuracy": a, "mean_token_cost": c, "mean_rollouts": u,
                        "stop_distribution": {int(k): float(v) for k, v in r.stop_n.value_counts(normalize=True).sort_index().items()},
                        "incorrect_early_stops_frac": float(len(early_wrong) / len(r)),
                        "incorrect_early_stops_recoverable_by_sc32_frac": float(early_wrong.sc32_correct.mean()) if len(early_wrong) else 0.0,
                        "by_pool_correct_fraction": r.assign(bin=pd.cut(r.pool_frac_correct, [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0])).groupby("bin", observed=True)
                        .agg(mean_rollouts=("stop_n", "mean"), accuracy=("correct", "mean"), n=("correct", "size")).reset_index().astype({"bin": str}).to_dict("records")},
        }
        r.to_parquet(f"{OUT}/adaptive_{ds}_{fam}_heldout.parquet")
    fixed = {}
    for N in BUDGETS:
        acc, cost = [], []
        for sid in ev:
            q = Q[sid]
            acc.append(np.mean([q["group_correct"][stats[(sid, o, N)]["sc"]] for o in range(N_ORDERS)]))
            cost.append(np.mean([int(q["T"][np.asarray(ORDERS[ds][sid][o][:N])].sum()) for o in range(N_ORDERS)]))
        fixed[N] = {"accuracy": float(np.mean(acc)), "mean_token_cost": float(np.mean(cost))}
    out["heldout_fixed_sc"] = fixed
    return out


ORDERS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="localonly/pa3_data")
    ap.add_argument("--meta", default="localonly/pa3_data/mathvista_meta/testmini.json")
    ap.add_argument("--datasets", default="math,mathvista")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    splits_all = json.load(open("data/task3/splits.json"))
    summary, sig_all, ad_all, prep_all, ex_all = {}, {}, {}, {}, {}
    sels = []
    for ds in a.datasets.split(","):
        t0 = time.time()
        df, checks, meta = load(a.data, ds, a.meta)
        log(f"{ds}: loaded {checks}")
        Q, gm = prepare(df, ds, meta)
        del df
        splits = {"dev": set(splits_all[ds]["dev"]), "eval": splits_all[ds]["eval"]}
        rng = np.random.default_rng(SEED)
        ORDERS[ds] = {sid: [rng.permutation(32).tolist() for _ in range(N_ORDERS)] for sid in sorted(Q)}
        lab_counts = {"no_answer_rollouts": int(sum(l == NO_ANSWER for q in Q.values() for l in q["labels"])),
                      "invalid_rollouts": int(sum(l == INVALID for q in Q.values() for l in q["labels"])),
                      "rollout_accuracy": float(np.mean([q["correct"].mean() for q in Q.values()])),
                      "mean_groups_per_question_32": float(np.mean([len(set(q["labels"])) for q in Q.values()]))}
        prep_all[ds] = {"checks": checks, "gold_mismatch_vs_metadata": gm, "scorer": SCORER[ds], "label_counts": lab_counts,
                        "groups": {sid: {"labels": q["labels"], "group_correct": {k: bool(v) for k, v in q["group_correct"].items()}, "reps": q["reps"]} for sid, q in Q.items()}}
        log(f"{ds}: prepared in {time.time() - t0:.0f}s {lab_counts}")
        sel, stats = run_dataset(ds, Q, {"dev": splits["dev"], "eval": set(splits["eval"])}, ORDERS[ds])
        sels.append(sel)
        summary[ds] = summarize(sel, ds, np.random.default_rng(SEED))
        sig_all[ds], ex_all[ds] = signals(Q, ds, splits)
        ad_all[ds] = adaptive(Q, stats, ds, {"dev": sorted(splits["dev"]), "eval": splits["eval"]})
        log(f"{ds}: done in {time.time() - t0:.0f}s; eval SC@32={summary[ds]['methods']['eval']['sc'][32]['accuracy']:.4f} "
            f"oracle@32={summary[ds]['methods']['eval']['sc'][32]['oracle_coverage']:.4f}")
    pd.concat(sels).to_parquet(f"{OUT}/selections.parquet")
    for name, obj in [("summary", summary), ("signals", sig_all), ("adaptive", ad_all), ("prepared", prep_all), ("illustrative_trajectories", ex_all), ("orders", ORDERS)]:
        with open(f"{OUT}/{name}.json", "w") as f:
            json.dump(obj, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log("offline analysis written to " + OUT)


if __name__ == "__main__":
    main()
