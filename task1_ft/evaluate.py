"""Task 1 evaluation of one condition: lm-eval benchmarks + per-example outputs + the 24 qualitative prompts.

python -m task1_ft.evaluate --cond baseline|fullft_mid|fullft|l2sp|lora4|lora16|oplora [--limit N]
Adapters are merged into a fresh fp16 backbone (projected factors for OPLoRA) after a merged-vs-unmerged logit check.
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file

from common import lora as L
from common.utils import SEED, env_info, gpu_mem, jdump, jload, jsonl_write, load_causal_lm, log, sync, chat_prompt
from task1_ft.data import MODEL, get_tok, probe_prompts, qualitative_prompts

RES = "results/task1"
CONDS = {
    "baseline": None,
    "fullft_mid": ("full", "fullft/mid"),
    "fullft": ("full", "fullft/final"),
    "l2sp": ("full", "l2sp/final"),
    "lora4": ("adapter", "lora4/final"),
    "lora16": ("adapter", "lora16/final"),
    "oplora": ("adapter", "oplora/final"),
}
GSM_MAX_NEW = 1024
QUAL_MAX_NEW = 512


def build_model(cond, out_root, tok, dev):
    info = {"cond": cond}
    model = load_causal_lm(MODEL, torch.float16).to(dev)
    spec = CONDS[cond]
    if spec is None:
        pass
    elif spec[0] == "full":
        sd = load_file(os.path.join(out_root, spec[1], "model_fp16.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all(k == "lm_head.weight" for k in missing), missing
        if missing:
            model.tie_weights()
        info["source"] = spec[1]
    else:
        d = os.path.join(out_root, spec[1])
        ex = jload(os.path.join(d, "export_info.json"))
        fac = load_file(os.path.join(d, "adapter.safetensors"))
        # unmerged reference: fresh backbone + exported factors as live adapters
        unm = L.attach_fixed_factors(load_causal_lm(MODEL, torch.float16).to(dev), fac, ex["r"], ex["alpha"]).eval()
        t0 = time.time()
        n = L.merge_factors(model, fac, ex["scaling"])
        info["merge_time_s"] = time.time() - t0
        info["merged_modules"] = n
        model.eval()
        prompts = probe_prompts(tok, jload(f"{RES}/probe_set.json")["probes"][:2] + jload(f"{RES}/probe_set.json")["probes"][32:34])
        diffs, agree = [], []
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
                a = unm(ids).logits.float()
                b = model(ids).logits.float()
                diffs.append(float((a - b).abs().max()))
                agree.append(float((a.argmax(-1) == b.argmax(-1)).float().mean()))
        info["merge_check"] = {"max_abs_logit_diff": max(diffs), "per_prompt_max_abs": diffs, "top1_agreement": sum(agree) / len(agree), "n_prompts": len(prompts), "dtype": "fp16"}
        log(f"merge check {cond}: {info['merge_check']}")
        del unm
        torch.cuda.empty_cache()
        info["source"] = spec[1]
    model.eval()
    model.config.use_cache = True
    return model, info


def make_lm(model, tok, batch_size):
    from lm_eval.models.huggingface import HFLM

    class RecordingHFLM(HFLM):
        """records the raw generated token count per prompt so truncation can be flagged exactly"""

        gen_records = {}

        def _model_generate(self, context, max_length, stop, **kw):
            out = super()._model_generate(context, max_length, stop, **kw)
            am = kw.get("attention_mask")
            eos = {self.tokenizer.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]}
            L0 = context.shape[1]
            for row in range(out.shape[0]):
                ctx = context[row][am[row].bool()] if am is not None else context[row]
                gen = out[row, L0:].tolist()
                n_until_eos = next((i for i, t in enumerate(gen) if t in eos), None)
                n = len(gen) if n_until_eos is None else n_until_eos
                txt = self.tokenizer.decode(gen[:n], skip_special_tokens=True)
                hit_stop = any(s in txt for s in stop if s not in ("<|im_end|>", "<|endoftext|>", "</s>"))
                self.gen_records[tuple(ctx.tolist())] = {"n_gen_tokens_raw": n, "hit_eos": n_until_eos is not None, "hit_stop_string": hit_stop,
                                                         "cap": max_length - L0, "truncated": (n_until_eos is None) and (not hit_stop) and n >= max_length - L0}
            return out

    return RecordingHFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, enable_thinking=False, max_length=4096)


def run_lm_eval(model, tok, cond, limit, timing):
    import lm_eval

    seeds = dict(random_seed=SEED, numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED)
    res = {}
    # GSM8K: zero-shot gsm8k_cot, non-thinking chat template, greedy, 1024 new tokens, stock stops/filters
    t0 = time.time()
    for bs in (64, 32):  # 64 for every condition; 32 only as an OOM fallback (recorded)
        lm = make_lm(model, tok, bs)
        try:
            r = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k_cot"], num_fewshot=0, apply_chat_template=True, fewshot_as_multiturn=False,
                                        gen_kwargs={"max_gen_toks": GSM_MAX_NEW, "do_sample": False}, log_samples=True, limit=limit, **seeds)
            break
        except torch.OutOfMemoryError:
            log(f"{cond}: gsm8k OOM at batch {bs}, retrying smaller")
            del lm
            torch.cuda.empty_cache()
            if bs == 32:
                raise
    timing["gsm8k_batch_size"] = bs
    sync()
    timing["gsm8k_s"] = time.time() - t0
    res["gsm8k_cot"] = r["results"]["gsm8k_cot"]
    res["configs"] = {"gsm8k_cot": r["configs"]["gsm8k_cot"]}
    jsonl_write(r["samples"]["gsm8k_cot"], f"{RES}/samples/raw/{cond}_gsm8k_cot.jsonl")  # raw first, parsed below
    try:
        gsm_rows = []
        by_doc = {}
        for s in r["samples"]["gsm8k_cot"]:
            by_doc.setdefault(s["doc_id"], []).append(s)
        for doc_id, ss in sorted(by_doc.items()):
            s0 = ss[0]
            ctx = s0["arguments"][0][0] if isinstance(s0["arguments"][0], (list, tuple)) else s0["arguments"]["gen_args_0"]["arg_0"]
            resp = s0["resps"][0][0] if isinstance(s0["resps"][0], list) else s0["resps"][0]
            row = {"doc_id": doc_id, "question": s0["doc"]["question"], "target": s0["target"], "response": resp}
            for s in ss:
                f = s.get("filter")
                row[f"extracted_{f}"] = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
                row[f"exact_match_{f}"] = s.get("exact_match")
            rec = lm.gen_records.get(tuple(tok(ctx, add_special_tokens=False).input_ids))
            if rec:
                row.update(rec)  # truncated = no EOS and no stop string within the 1024-token cap
            row["n_resp_tokens"] = len(tok(resp, add_special_tokens=False).input_ids)
            row["extractable_flexible"] = row.get("extracted_flexible-extract") not in (None, "", "[invalid]")
            gsm_rows.append(row)
        jsonl_write(gsm_rows, f"{RES}/samples/{cond}_gsm8k.jsonl")
        n = len(gsm_rows)
        res["gsm8k_extra"] = {
            "n": n,
            "truncation_rate": sum(bool(x.get("truncated")) for x in gsm_rows) / n,
            "flexible_unextractable_rate": sum(not x["extractable_flexible"] for x in gsm_rows) / n,
            "strict_invalid_rate": sum(x.get("extracted_strict-match") in (None, "", "[invalid]") for x in gsm_rows) / n,
            "mean_resp_tokens": sum(x["n_resp_tokens"] for x in gsm_rows) / n,
            "records_matched": sum("n_gen_tokens_raw" in x for x in gsm_rows),
        }
        log(f"{cond} gsm8k: {res['gsm8k_cot']} {res['gsm8k_extra']}")
    except Exception as e:  # never lose an eval to a log-format surprise; raw samples are already saved
        log(f"gsm8k sample parsing failed: {e!r}")
        res["gsm8k_extra"] = {"parse_error": repr(e)}

    # likelihood tasks: stock completion prompts, no chat template, zero-shot
    lm2 = make_lm(model, tok, 16)
    t0 = time.time()
    r2 = lm_eval.simple_evaluate(model=lm2, tasks=["hellaswag", "boolq", "arc_easy"], num_fewshot=0, apply_chat_template=False,
                                 log_samples=True, limit=limit, **seeds)
    sync()
    timing["likelihood_tasks_s"] = time.time() - t0
    for t in ["hellaswag", "boolq", "arc_easy"]:
        jsonl_write(r2["samples"][t], f"{RES}/samples/raw/{cond}_{t}.jsonl")
        res[t] = r2["results"][t]
        res["configs"][t] = r2["configs"][t]
        try:
            rows = []
            for s in r2["samples"][t]:
                ll = [x[0] if isinstance(x, (list, tuple)) else x for x in s["filtered_resps"]]
                rows.append({"doc_id": s["doc_id"], "target": s["target"], "loglik": ll, "acc": s.get("acc"), "acc_norm": s.get("acc_norm")})
            jsonl_write(rows, f"{RES}/samples/{cond}_{t}.jsonl")
        except Exception as e:
            log(f"{t} sample parsing failed: {e!r}")
    res["versions"] = {**r.get("versions", {}), **r2.get("versions", {})}
    res["n_samples"] = {**r.get("n-samples", {}), **r2.get("n-samples", {})}
    log(f"{cond} likelihood: " + json.dumps({t: res[t] for t in ['hellaswag', 'boolq', 'arc_easy']}, default=str))
    return res


@torch.no_grad()
def qualitative(model, tok, cond, dev):
    qs = qualitative_prompts()
    texts = []
    for q in qs:
        msgs = ([{"role": "system", "content": q["system"]}] if q["system"] else []) + [{"role": "user", "content": q["user"]}]
        texts.append(chat_prompt(tok, msgs, enable_thinking=False))
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=QUAL_MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
    sync()
    dt = time.time() - t0
    tok.padding_side = "right"
    eos = {tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"]}
    rows = []
    for q, prompt, seq in zip(qs, texts, out):
        gen = seq[enc.input_ids.shape[1]:].tolist()
        n = next((i for i, t in enumerate(gen) if t in eos), None)
        rows.append({**q, "cond": cond, "prompt": prompt, "response": tok.decode(gen[: n if n is not None else len(gen)], skip_special_tokens=True),
                     "n_tokens": n if n is not None else len(gen), "truncated": n is None})
    jdump({"cond": cond, "decoding": "greedy, batch 24 left-padded, fp16, 512 new tokens, thinking disabled", "gen_time_s": dt, "responses": rows},
          f"{RES}/qualitative/{cond}.json")
    log(f"{cond}: qualitative done in {dt:.0f}s, truncated {sum(r['truncated'] for r in rows)}/24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS))
    ap.add_argument("--out-root", default="outputs/task1")
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tok = get_tok()
    tok.padding_side = "right"
    t0 = time.time()
    model, info = build_model(a.cond, a.out_root, tok, dev)
    timing = {"load_merge_s": time.time() - t0}
    res = run_lm_eval(model, tok, a.cond, a.limit, timing)
    t1 = time.time()
    qualitative(model, tok, a.cond, dev)
    timing["qualitative_s"] = time.time() - t1
    primary = {
        "gsm8k_flexible_em": res["gsm8k_cot"].get("exact_match,flexible-extract"),
        "gsm8k_strict_em": res["gsm8k_cot"].get("exact_match,strict-match"),
        "hellaswag_acc_norm": res["hellaswag"].get("acc_norm,none"), "hellaswag_acc": res["hellaswag"].get("acc,none"),
        "boolq_acc": res["boolq"].get("acc,none"),
        "arc_easy_acc_norm": res["arc_easy"].get("acc_norm,none"), "arc_easy_acc": res["arc_easy"].get("acc,none"),
    }
    out = {"cond": a.cond, "primary": primary, "limit": a.limit, "build": info, "timing_s": timing, "gpu": gpu_mem(), "lm_eval": res,
           "settings": {"dtype": "fp16", "num_fewshot": 0, "gsm8k": "gsm8k_cot, non-thinking chat template, greedy, max_gen_toks 1024, stock until/filters",
                        "likelihood": "stock prompts, no chat template", "batch_size": {"gsm8k": 64, "likelihood": 16}}, "env": env_info()}
    jdump(out, f"{RES}/eval_{a.cond}.json")
    log(f"{a.cond} primary: {primary}")


if __name__ == "__main__":
    main()
