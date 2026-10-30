"""Dev/eval split for Task 3: sort the 1,000 question IDs, permute with default_rng(6304), first 200 = dev.
The first 50 MATH eval questions in that permutation form the live-generation subset.
python -m task3_ttc.make_splits --data <ATML-PA3-Data dir>
"""
import argparse

import numpy as np
import pandas as pd

import json
import os

SEED = 6304


def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w"), indent=2)


def jsonl_write(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="localonly/pa3_data")
    a = ap.parse_args()
    out = {"seed": SEED, "rule": "sorted(ids) -> np.random.default_rng(6304).permutation -> [:200] dev, [200:] eval"}
    for ds, folder in FOLDERS.items():
        g = pd.read_parquet(f"{a.data}/{folder}/generations.parquet", columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
        ids = sorted(g["sample_id"].unique().tolist())
        assert len(ids) == 1000
        perm = [ids[i] for i in np.random.default_rng(SEED).permutation(len(ids))]
        out[ds] = {"dev": perm[:200], "eval": perm[200:]}
        if ds == "math":
            live = perm[200:250]
            out["math_live"] = live
            q = g[g.rollout_id == 0].set_index("sample_id")
            jsonl_write([{"sample_id": s, "question_text": q.loc[s, "question_text"], "gold_answer": q.loc[s, "gold_answer"]} for s in live],
                        "data/task3/math_live_questions.jsonl")
    jdump(out, "data/task3/splits.json")
    print({k: (len(v["dev"]), len(v["eval"])) for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
