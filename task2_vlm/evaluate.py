"""Task 2 final evaluation on both final-test suites under correct / complementary / mismatched / no-image inputs.

python -m task2_vlm.evaluate --run main [--pool]
Greedy, thinking disabled, 16 new tokens, batch 8. References stay the original ones in every condition.
Grounding: normalised exact match (presence / counting). VQAv2: local soft consensus accuracy (yes/no, number, other).
"""
import argparse

import numpy as np
import torch
from safetensors.torch import load_file

from common import lora as L
from common.utils import SEED, env_info, gpu_mem, jdump, jload, jsonl_write, load_causal_lm, log
from task2_vlm.train import DATA, LORA_A, LORA_DROP, LORA_R, ex_from_pair
from task2_vlm.vlm import LLM, Builder, Connector, Features, greedy, per_example_ce
from task2_vlm.vqa_norm import normalize, vqa_soft_acc

CONDS = ["correct", "complementary", "mismatched", "noimage"]
N_BOOT = 4000


def image_for(cond, p, s, by_id, mm):
    if cond == "correct":
        return p["sides"][s]["image"]
    if cond == "complementary":
        return p["sides"][1 - s]["image"]
    if cond == "mismatched":
        return by_id[mm[p["pair_id"]]]["sides"][s]["image"]
    return None


def score(suite, p, s, pred):
    side = p["sides"][s]
    if suite == "vqa":
        return vqa_soft_acc(pred, side["answers"])
    return float(normalize(pred) == normalize(side["canonical"]))


def boot_ci(x, rng, n=N_BOOT):
    """x: per-pair values (n_pairs,), resampled as complete pairs"""
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--pool", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    from task1_ft.data import get_tok
    tok = get_tok()
    feats = Features("outputs/task2/features", pool=a.pool)
    bld = Builder(tok, feats.n_vis)
    llm = load_causal_lm(LLM, torch.float16).to(dev)
    L.inject_lora(llm, L.T2_TARGETS, LORA_R, LORA_A, LORA_DROP, SEED)
    L.load_raw_state(llm, load_file(f"outputs/task2/{a.run}/stage3/lora.safetensors"))
    conn = Connector().to(dev)
    conn.load_state_dict({k: v.to(dev) for k, v in load_file(f"outputs/task2/{a.run}/stage3/connector.safetensors").items()})
    llm.eval()
    conn.eval()
    mm_all = jload(f"{DATA}/mismatch_map.json")
    rng = np.random.default_rng(SEED)
    summary = {"run": a.run, "n_visual_tokens": feats.n_vis, "settings": "greedy, 16 new tokens, batch 8, thinking disabled, fp16", "suites": {}}
    rows_all = []
    for suite, fname, mkey in [("grounding", "grounding_test", "grounding_test"), ("vqa", "vqa_test", "vqa_test")]:
        pairs = jload(f"{DATA}/{fname}.json")
        by_id = {p["pair_id"]: p for p in pairs}
        mm = mm_all[mkey]
        per = {}  # cond -> dict of arrays aligned to (pair, side)
        for cond in CONDS:
            exs = []
            for p in pairs:
                for s in (0, 1):
                    exs.append(dict(ex_from_pair(p, s), image=image_for(cond, p, s, by_id, mm)))
            preds, gen_t = [], 0.0
            for i in range(0, len(exs), 8):
                t, dt = greedy(llm, conn, feats, bld, exs[i : i + 8], dev)
                preds += t
                gen_t += dt
            nll = []
            with torch.no_grad():
                for i in range(0, len(exs), 8):
                    b = bld.batch(exs[i : i + 8], dev)
                    with torch.autocast("cuda", dtype=torch.float16):
                        _, n = per_example_ce(llm, conn, feats, b, dev)
                    nll += n.tolist()
            acc = np.array([score(suite, p, s, preds[2 * k + s]) for k, p in enumerate(pairs) for s in (0, 1)])
            canon_hit = np.array([normalize(preds[2 * k + s]) == normalize(p["sides"][s]["canonical"]) for k, p in enumerate(pairs) for s in (0, 1)])
            follow = np.array([normalize(preds[2 * k + s]) == normalize(p["sides"][1 - s]["canonical"]) for k, p in enumerate(pairs) for s in (0, 1)])
            per[cond] = {"preds": preds, "acc": acc, "canon": canon_hit, "follow_partner": follow, "nll": np.array(nll), "gen_s": gen_t}
            for k, p in enumerate(pairs):
                for s in (0, 1):
                    j = 2 * k + s
                    rows_all.append({"suite": suite, "cond": cond, "pair_id": p["pair_id"], "side": s, "type": p["type"], "question": p["question"],
                                     "image": exs[j]["image"], "reference": p["sides"][s]["canonical"], "target": p["sides"][s]["target"],
                                     "answers": p["sides"][s].get("answers"), "partner_reference": p["sides"][1 - s]["canonical"],
                                     "prediction": preds[j], "prediction_norm": normalize(preds[j]), "score": float(acc[j]),
                                     "nll": nll[j], "follows_partner": bool(follow[j])})
            log(f"{suite}/{cond}: acc={acc.mean():.3f} nll={np.mean(nll):.3f}")

        types = np.array([p["type"] for p in pairs for _ in (0, 1)])
        S = {}
        for cond in CONDS:
            r = per[cond]
            pair_acc = (r["canon"][0::2] & r["canon"][1::2]).astype(float)
            pair_mean = (r["acc"][0::2] + r["acc"][1::2]) / 2
            same = np.array([normalize(r["preds"][2 * k]) == normalize(r["preds"][2 * k + 1]) for k in range(len(pairs))], float)
            S[cond] = {
                "accuracy": float(r["acc"].mean()), "accuracy_ci95": boot_ci(pair_mean, rng),
                "by_type": {t: float(r["acc"][types == t].mean()) for t in sorted(set(types))},
                "pair_accuracy": float(pair_acc.mean()), "pair_accuracy_ci95": boot_ci(pair_acc, rng),
                "identical_answer_rate": float(same.mean()), "empty_answer_rate": float(np.mean([normalize(x) == "" for x in r["preds"]])),
                "mean_ref_nll": float(r["nll"].mean()), "gen_time_per_answer_s": r["gen_s"] / len(r["preds"]),
                "follows_partner_answer_rate": float(r["follow_partner"].mean()),
            }
        # correct-image advantage over complementary swap (paired over the same bootstrap pairs)
        c, w = per["correct"], per["complementary"]
        d_acc = (c["acc"][0::2] + c["acc"][1::2]) / 2 - (w["acc"][0::2] + w["acc"][1::2]) / 2
        d_nll = (w["nll"][0::2] + w["nll"][1::2]) / 2 - (c["nll"][0::2] + c["nll"][1::2]) / 2
        idx = rng.integers(0, len(pairs), size=(N_BOOT, len(pairs)))
        S["advantage_correct_vs_complementary"] = {
            "accuracy_pp": float(100 * d_acc.mean()), "accuracy_pp_ci95": [float(100 * np.percentile(d_acc[idx].mean(1), q)) for q in (2.5, 97.5)],
            "nll_swap_penalty": float(d_nll.mean()), "nll_swap_penalty_ci95": [float(np.percentile(d_nll[idx].mean(1), q)) for q in (2.5, 97.5)],
            "by_type_accuracy_pp": {t: float(100 * (c["acc"][types == t].mean() - w["acc"][types == t].mean())) for t in sorted(set(types))},
        }
        S["n_pairs"] = len(pairs)
        S["n_questions"] = 2 * len(pairs)
        summary["suites"][suite] = S
    summary["gpu"] = gpu_mem()
    summary["env"] = env_info()
    summary["bootstrap"] = {"resamples": N_BOOT, "unit": "complete pairs", "seed": SEED, "paired_across_conditions": "advantage uses one shared resample set"}
    jsonl_write(rows_all, f"results/task2/{a.run}/final_test_predictions.jsonl")
    jdump(summary, f"results/task2/{a.run}/final_eval.json")
    log(f"task2 {a.run} final: " + str({s: {c: summary['suites'][s][c]['accuracy'] for c in CONDS} for s in summary['suites']}))


if __name__ == "__main__":
    main()
