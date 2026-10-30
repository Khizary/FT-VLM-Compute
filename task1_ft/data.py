"""Task 1 data: first 40k eligible MetaMathQA examples (source order), probe set, qualitative prompts.

python -m task1_ft.data --out outputs/task1/data
"""
import argparse
import glob
import os
import re
import time
import unicodedata
from collections import Counter

import numpy as np
import torch
import yaml

from common.utils import SEED, chat_prompt, jdump, log

MODEL = "Qwen/Qwen3-0.6B"
N_TRAIN = 40_000
MAX_LEN = 2048
RES = "results/task1"


def get_tok():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def norm_text(s):
    """GSM8K matching normalisation: NFKC, lowercase, non-alphanumerics -> space, collapse whitespace"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def user_prompt(tok, q):
    return chat_prompt(tok, [{"role": "user", "content": q}], enable_thinking=False)


def build(out_dir):
    from datasets import load_dataset

    t0 = time.time()
    tok = get_tok()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_norm = {norm_text(q) for q in gsm["question"]}
    mm = load_dataset("meta-math/MetaMathQA", split="train")
    log(f"MetaMathQA rows: {len(mm)}; gsm8k test: {len(gsm)}")

    # sanity: the template we train on must equal (generation prompt + response + <|im_end|>)
    demo = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}], tokenize=False, enable_thinking=False
    )
    gen = user_prompt(tok, "hi")
    if not demo.startswith(gen + "ok<|im_end|>"):
        log(f"WARNING chat template mismatch: {demo!r} vs {gen!r} (we train on gen prompt + response)")

    stats = Counter()
    src_idx, prompts, resps, types = [], [], [], []
    for i, ex in enumerate(mm):
        q, r = ex["query"], ex["response"]
        if not q or not q.strip() or not r or not r.strip():
            stats["empty"] += 1
            continue
        if norm_text(q) in gsm_norm or norm_text(ex.get("original_question")) in gsm_norm:
            stats["gsm8k_test_match"] += 1
            continue
        p_ids = tok(user_prompt(tok, q), add_special_tokens=False).input_ids
        r_ids = tok(r, add_special_tokens=False).input_ids + [im_end]
        if len(p_ids) + len(r_ids) > MAX_LEN:
            stats["too_long"] += 1
            continue
        src_idx.append(i)
        prompts.append(p_ids)
        resps.append(r_ids)
        types.append(ex.get("type"))
        if len(src_idx) == N_TRAIN:
            break
    assert len(src_idx) == N_TRAIN, len(src_idx)
    stats["scanned"] = src_idx[-1] + 1
    stats["kept"] = len(src_idx)

    os.makedirs(out_dir, exist_ok=True)
    torch.save({"src_idx": src_idx, "prompt_ids": prompts, "resp_ids": resps}, os.path.join(out_dir, "train_tok.pt"))
    lens = np.array([len(p) + len(r) for p, r in zip(prompts, resps)])
    rlens = np.array([len(r) for r in resps])
    summary = {
        "filter_counts": dict(stats),
        "type_counts": dict(Counter(types)),
        "formatted_len": {"mean": float(lens.mean()), "p50": float(np.median(lens)), "p99": float(np.percentile(lens, 99)), "max": int(lens.max())},
        "response_tokens_total": int(rlens.sum()),
        "all_tokens_total": int(lens.sum()),
        "normalisation": "NFKC, lowercase, [^a-z0-9]+ -> space, collapse whitespace; drop if query or original_question matches a GSM8K test question",
        "build_time_s": time.time() - t0,
    }
    jdump(summary, f"{RES}/data_summary.json")
    jdump({"metamathqa_source_indices": src_idx}, f"{RES}/train_example_ids.json")
    log(f"data summary: {summary}")
    make_probe_set()


def hs_preprocess(text):
    # identical to lm-eval's hellaswag preprocessing
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def make_probe_set():
    """32 GSM8K test questions + 32 HellaSwag val contexts, seed 6304; fixed before any results exist"""
    from datasets import load_dataset

    path = f"{RES}/probe_set.json"
    if os.path.exists(path):
        return
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    rng = np.random.default_rng(SEED)
    gi = sorted(rng.choice(len(gsm), 32, replace=False).tolist())
    hi = sorted(rng.choice(len(hs), 32, replace=False).tolist())
    probes = []
    for i in gi:
        probes.append({"set": "math", "source": "gsm8k_test", "idx": i, "text": gsm[i]["question"]})
    for i in hi:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        probes.append({"set": "nonmath", "source": "hellaswag_validation", "idx": i, "ind": d.get("ind"), "text": hs_preprocess(d["activity_label"] + ": " + ctx)})
    jdump({"seed": SEED, "probes": probes}, path)


def probe_prompts(tok, probes):
    """benchmark prompt formats: gsm8k_cot zero-shot under the non-thinking chat template; hellaswag raw context"""
    out = []
    for p in probes:
        if p["set"] == "math":
            out.append(user_prompt(tok, f"Q: {p['text']}\nA:"))
        else:
            out.append(p["text"])
    return out


def qualitative_prompts():
    d = yaml.safe_load(open("data/task1/first_plot_questions.yaml"))
    out = []
    for q in d:
        qid = q["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid[: -len("_" + fmt)] if fmt != "plain" else qid
        assert len(q["paraphrases"]) == 1
        out.append({"id": qid, "question": base, "format": fmt, "system": q.get("system"), "user": q["paraphrases"][0]})
    assert len(out) == 24
    return out


def orig_weights():
    """starting checkpoint tensors exactly as stored on the hub (bf16), keyed by parameter name"""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    d = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/task1/data")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "train_tok.pt")):
        log("train data already built")
        make_probe_set()
    else:
        build(a.out)
