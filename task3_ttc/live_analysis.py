"""Task 3 Step 4 analysis: single long response vs parallel x3 (self-consistency) vs sequential refinement.

python -m task3_ttc.live_analysis
Costs are actual: refinement is charged the shared initial response (parallel[0]) plus the critique/revision call,
whose input includes the longer revision prompt. Transitions are relative to the shared initial answer.
"""
import json

import numpy as np

from task3_ttc.answers import NO_ANSWER, SCORER, group_math

SEED = 6304
IN = "results/task3/live_generations.jsonl"
OUT = "results/task3/live_summary.json"


def main():
    rows = [json.loads(l) for l in open(IN)]
    assert len(rows) == 50, len(rows)
    per = []
    for r in rows:
        gold = r["gold_answer"]
        single, par, ref = r["single"], r["parallel"], r["refine"]
        answers = [single["extracted"]] + [p["extracted"] for p in par] + [ref["extracted"]]
        labels, corr, reps = group_math(answers, gold)  # one equivalence pass per question; gold only grades
        l_single, l_par, l_ref = labels[0], labels[1:4], labels[4]
        tot, first = {}, {}
        for i, l in enumerate(l_par):
            tot[l] = tot.get(l, 0) + 1
            first.setdefault(l, i)
        best = max(tot.values())
        sc = min((l for l in tot if tot[l] == best), key=lambda l: first[l])  # ties -> earliest sample
        calls = {"single": [single], "parallel": par, "refinement": [par[0], ref]}
        e = {"sample_id": r["sample_id"], "gold": gold,
             "correct": {"single": corr[l_single], "parallel_sc": corr[sc], "refinement": corr[l_ref], "initial": corr[l_par[0]],
                         "parallel_any": any(corr[l] for l in l_par)},
             "answers": {"single": single["extracted"], "parallel": [p["extracted"] for p in par], "parallel_sc_group": reps.get(sc, NO_ANSWER),
                         "initial": par[0]["extracted"], "revised": ref["extracted"]},
             "parallel_unanimous": len(set(l_par)) == 1, "revision_changed_answer": l_ref != l_par[0]}
        for k, cs in calls.items():
            e[k] = {"input_tokens": sum(c["input_tokens"] for c in cs), "output_tokens": sum(c["output_tokens"] for c in cs),
                    "time_s": sum(c["time_s"] for c in cs), "n_calls": len(cs),
                    "truncated_calls": sum(c["truncated"] for c in cs), "extracted_calls": sum(c["extracted"] is not None for c in cs)}
        e["final_call_truncated"] = {"single": single["truncated"], "refinement": ref["truncated"]}
        e["final_extracted"] = {"single": single["extracted"] is not None, "refinement": ref["extracted"] is not None,
                                "parallel_sc": l_par and sc != NO_ANSWER}
        per.append(e)

    def mean(f):
        return float(np.mean([f(e) for e in per]))

    strat = {}
    for k, ck in [("single", "single"), ("parallel", "parallel_sc"), ("refinement", "refinement")]:
        strat[k] = {"accuracy": mean(lambda e: e["correct"][ck]),
                    "mean_input_tokens": mean(lambda e: e[k]["input_tokens"]), "mean_output_tokens": mean(lambda e: e[k]["output_tokens"]),
                    "mean_total_tokens": mean(lambda e: e[k]["input_tokens"] + e[k]["output_tokens"]),
                    "mean_time_s": mean(lambda e: e[k]["time_s"]), "total_time_min": float(sum(e[k]["time_s"] for e in per) / 60),
                    "output_allowance": 3072, "call_truncation_rate": float(sum(e[k]["truncated_calls"] for e in per) / sum(e[k]["n_calls"] for e in per)),
                    "call_extraction_rate": float(sum(e[k]["extracted_calls"] for e in per) / sum(e[k]["n_calls"] for e in per)),
                    "final_answer_extracted_rate": mean(lambda e: e["final_extracted"][ck])}
    strat["single"]["final_truncation_rate"] = mean(lambda e: e["final_call_truncated"]["single"])
    strat["refinement"]["final_truncation_rate"] = mean(lambda e: e["final_call_truncated"]["refinement"])
    strat["parallel"]["oracle_any_of_3"] = mean(lambda e: e["correct"]["parallel_any"])
    strat["parallel"]["unanimous_rate"] = mean(lambda e: e["parallel_unanimous"])
    strat["initial_response_only"] = {"accuracy": mean(lambda e: e["correct"]["initial"])}
    trans = {"wrong_to_correct": sum((not e["correct"]["initial"]) and e["correct"]["refinement"] for e in per),
             "correct_to_wrong": sum(e["correct"]["initial"] and not e["correct"]["refinement"] for e in per),
             "correct_to_correct": sum(e["correct"]["initial"] and e["correct"]["refinement"] for e in per),
             "wrong_to_wrong": sum((not e["correct"]["initial"]) and not e["correct"]["refinement"] for e in per),
             "answer_changed": sum(e["revision_changed_answer"] for e in per)}
    rng = np.random.default_rng(SEED)
    acc = {k: np.array([float(e["correct"][c]) for e in per]) for k, c in [("single", "single"), ("parallel", "parallel_sc"), ("refinement", "refinement")]}
    idx = rng.integers(0, 50, size=(1000, 50))
    boot = {}
    for a, b in [("parallel", "single"), ("refinement", "single"), ("refinement", "parallel")]:
        d = acc[a] - acc[b]
        boot[f"{a}-{b}"] = {"diff_pp": float(100 * d.mean()), "ci95_pp": [float(100 * np.percentile(d[idx].mean(1), q)) for q in (2.5, 97.5)]}
    meta = json.load(open("results/task3/live_meta.json"))
    out = {"n_questions": 50, "scorer": SCORER["math"], "strategies": strat, "refinement_transitions": trans, "paired_bootstrap_1000": boot,
           "generation": {k: meta[k] for k in ["model", "revision", "dtype", "batch_size", "sampling"]}, "peak_mem": meta["peak_mem_gb"],
           "note": "time excludes model loading; ran on GPU1 of a T4x2 while GPU0 trained Task 2 (strategies interleaved per question)",
           "per_question": per}
    json.dump(out, open(OUT, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({k: v for k, v in out.items() if k != "per_question"}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
