"""Task 3 Step 4: single long response vs 3 parallel samples vs sequential refinement on 50 MATH questions.

python -m task3_ttc.live_generate [--limit N]
Qwen2.5-3B-Instruct (pinned revision), fp16, batch size 1, T=0.7 top-p 0.8 top-k 20 rep 1.0.
Strategies are interleaved per question so all three see the same GPU conditions. Appends one JSON line per
question and skips questions already written (resumable). Grading / self-consistency happen offline with math-verify.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from common.utils import env_info, gpu_mem, jdump, jsonl_read, load_causal_lm, log, past_deadline, sync

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."
GEN = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
OUT = "results/task3/live_generations.jsonl"
SEED = 6304


def last_boxed(text):
    """README rule: last literal \\boxed{, scan balanced braces; absent / empty / unclosed -> None"""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                ans = text[start:j].strip()
                return ans or None
        j += 1
    return None


def call(model, tok, messages, cap, seed, dev):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sync()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=cap, pad_token_id=tok.eos_token_id, **GEN)
    sync()
    dt = time.time() - t0
    gen = out[0, ids.input_ids.shape[1]:].tolist()
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
    n = next((k for k, t in enumerate(gen) if t in eos), None)
    text = tok.decode(gen[: n if n is not None else len(gen)], skip_special_tokens=True)
    return {"prompt": prompt, "seed": seed, "cap": cap, "input_tokens": int(ids.input_ids.shape[1]),
            "output_tokens": n if n is not None else len(gen), "truncated": n is None, "time_s": dt,
            "response": text, "extracted": last_boxed(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    qs = jsonl_read("data/task3/math_live_questions.jsonl")[: a.limit]
    done = {r["sample_id"] for r in jsonl_read(OUT)} if os.path.exists(OUT) else set()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    t_load = time.time()
    model = load_causal_lm(MODEL, torch.float16, revision=REV).to(dev).eval()
    t_load = time.time() - t_load
    torch.cuda.reset_peak_memory_stats()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, "a")
    for qi, q in enumerate(qs):
        if q["sample_id"] in done:
            continue
        if past_deadline(margin=900):
            log("task3 live: deadline, stopping (rerun resumes)")
            break
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question_text"] + "\n\n" + SUFFIX}]
        s0 = SEED * 1000 + qi * 10  # independent seed per call, recorded
        single = call(model, tok, base, 3072, s0 + 0, dev)
        par = [call(model, tok, base, 1024, s0 + 1 + k, dev) for k in range(3)]
        rev_msgs = base + [{"role": "assistant", "content": par[0]["response"]}, {"role": "user", "content": REVIEW}]
        refine = call(model, tok, rev_msgs, 2048, s0 + 5, dev)  # initial response = first parallel sample
        row = {"q_index": qi, "sample_id": q["sample_id"], "gold_answer": q["gold_answer"], "single": single, "parallel": par,
               "refine_initial_ref": "parallel[0]", "refine": refine, "peak_mem_gb": gpu_mem()["peak_alloc_gb"]}
        f.write(json.dumps(row) + "\n")
        f.flush()
        log(f"live q{qi} {q['sample_id']}: single {single['output_tokens']}t/{single['time_s']:.0f}s, "
            f"parallel {[p['output_tokens'] for p in par]}, refine {refine['output_tokens']}t | gold {q['gold_answer']} "
            f"single={single['extracted']} par={[p['extracted'] for p in par]} rev={refine['extracted']}")
    f.close()
    jdump({"model": MODEL, "revision": REV, "dtype": "fp16", "batch_size": 1, "sampling": GEN, "model_load_s_excluded": t_load,
           "peak_mem_gb": gpu_mem(), "env": env_info(), "n_done": len(jsonl_read(OUT))}, "results/task3/live_meta.json")


if __name__ == "__main__":
    main()
