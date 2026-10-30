"""Task 2 curriculum: caption alignment (2,250) -> visual instruction tuning (1,500) -> natural VQA (750).

python -m task2_vlm.train --run main            (196 visual tokens)
python -m task2_vlm.train --run pool49 --pool   (optional 2x2-pooled 49 tokens)
Resumes from the newest checkpoint in outputs/task2/<run>/recovery.
"""
import argparse
import glob
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from safetensors.torch import save_file

from common import lora as L
from common.utils import (SEED, JsonlLog, env_info, gpu_mem, jdump, jload, load_causal_lm, log, past_deadline, path_size, rng_state,
                          seed_all, set_rng_state)
from task2_vlm.vlm import LLM, Builder, Connector, Features, per_example_ce

DATA = "results/task2/data"
MICRO, ACCUM = 8, 2
EFF = MICRO * ACCUM
STAGES = [
    {"name": "caption_alignment", "updates": 2250, "lr_c": 1e-3, "lr_l": None, "mix": None},
    {"name": "visual_instruction", "updates": 1500, "lr_c": 1e-4, "lr_l": 2e-5, "mix": [("presence", 0.5), ("counting", 0.25), ("caption", 0.25)]},
    {"name": "natural_vqa", "updates": 750, "lr_c": 5e-5, "lr_l": 1e-5, "mix": [("vqa", 0.7), ("presence", 0.1), ("counting", 0.1), ("caption", 0.1)]},
]
ENDS = np.cumsum([s["updates"] for s in STAGES]).tolist()  # 2250, 3750, 4500
TOTAL = ENDS[-1]
DEV_AT = sorted(set(range(0, TOTAL + 1, 500)) | set(ENDS))
CKPT_EVERY = 250
LORA_R, LORA_A, LORA_DROP = 16, 32, 0.05
EXIT_PREEMPT = 75


def stage_of(g):
    """stage index for global update g (1-based)"""
    return next(i for i, e in enumerate(ENDS) if g <= e)


def stage_lambda(U):
    W = math.ceil(0.05 * U)

    def f(i):
        u = i + 1
        if u <= W:
            return u / W
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (u - W) / (U - W))))
    return f


def ex_from_pair(p, side):
    s = p["sides"][side]
    kind = p["type"] if p["type"] in ("presence", "counting") else "vqa"
    return {"kind": kind, "question": p["question"], "target": s["target"], "image": s["image"], "pair_id": p["pair_id"], "side": side}


def cap_ex(c):
    return {"kind": "caption", "question": None, "target": c["caption"], "image": c["image"]}


def load_data():
    d = {k: jload(f"{DATA}/{k}.json") for k in ["captions_train", "captions_dev", "grounding_train_pool", "grounding_dev", "vqa_train", "vqa_dev"]}
    d["presence"] = [p for p in d["grounding_train_pool"] if p["type"] == "presence"]
    d["counting"] = [p for p in d["grounding_train_pool"] if p["type"] == "counting"]
    # split isolation re-check at train time
    tr = {c["image"] for c in d["captions_train"]} | {s["image"] for k in ["grounding_train_pool", "vqa_train"] for p in d[k] for s in p["sides"]}
    ev = {c["image"] for c in d["captions_dev"]} | {s["image"] for k in ["grounding_dev", "vqa_dev"] for p in d[k] for s in p["sides"]}
    for k in ["grounding_test", "vqa_test"]:
        ev |= {s["image"] for p in jload(f"{DATA}/{k}.json") for s in p["sides"]}
    assert not (tr & ev), "eval image leaked into training"
    return d


def draw_update(d, mix, rng, counts):
    """8 draws -> 16 examples. a question draw = one complete pair (both images); caption replay = two captions"""
    exs = []
    names, probs = zip(*mix)
    cum = np.cumsum(probs)
    for _ in range(EFF // 2):
        comp = names[int(np.searchsorted(cum, rng.random(), side="right"))]
        counts[comp] += 1
        if comp == "caption":
            for j in rng.integers(len(d["captions_train"]), size=2):
                exs.append(cap_ex(d["captions_train"][int(j)]))
        else:
            pool = d[comp] if comp != "vqa" else d["vqa_train"]
            p = pool[int(rng.integers(len(pool)))]
            exs += [ex_from_pair(p, 0), ex_from_pair(p, 1)]
    return exs


def stage1_examples(d, g):
    """two seeded passes over the 18,000 caption examples, each pass shuffled"""
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(d["captions_train"])) for _ in range(2)]
    out = []
    for pos in range((g - 1) * EFF, g * EFF):
        out.append(cap_ex(d["captions_train"][int(perms[pos // 18000][pos % 18000])]))
    return out


@torch.no_grad()
def dev_eval(llm, conn, feats, bld, d, dev):
    llm.eval()
    res = {}

    def run(exs):
        ce_all, nll_all = [], []
        for i in range(0, len(exs), 16):
            b = bld.batch(exs[i : i + 16], dev)
            with torch.autocast("cuda", dtype=torch.float16):
                ce, nll = per_example_ce(llm, conn, feats, b, dev)
            ce_all += ce.tolist()
            nll_all += nll.tolist()
        return np.array(ce_all), np.array(nll_all)

    ce, nll = run([cap_ex(c) for c in d["captions_dev"]])
    res["caption_loss"] = float(ce.mean())
    res["caption_nll_noterm"] = float(nll.mean())
    for suite in ["grounding_dev", "vqa_dev"]:
        pairs = d[suite]
        base = [ex_from_pair(p, s) for p in pairs for s in (0, 1)]
        partner = [dict(e, image=p["sides"][1 - s]["image"]) for e, (p, s) in zip(base, [(p, s) for p in pairs for s in (0, 1)])]
        noimg = [dict(e, image=None) for e in base]
        _, n_c = run(base)
        _, n_p = run(partner)
        _, n_n = run(noimg)
        r = {"nll_correct": float(n_c.mean()), "nll_complementary": float(n_p.mean()), "nll_noimage": float(n_n.mean()),
             "swap_penalty": float((n_p - n_c).mean()), "noimage_penalty": float((n_n - n_c).mean())}
        kinds = np.array([p["type"] for p in pairs for _ in (0, 1)])
        for k in sorted(set(kinds)):
            m = kinds == k
            r[f"{k}_nll_correct"] = float(n_c[m].mean())
            r[f"{k}_swap_penalty"] = float((n_p[m] - n_c[m]).mean())
        res[suite] = r
    llm.train()
    return res


@torch.no_grad()
def calibrate(llm, conn, feats, bld, d, dev):
    """rule: scale = mean L2 norm of Qwen text-token embeddings over training texts / mean L2 norm of the
    (unscaled, freshly initialised) connector outputs over training images. Training data only."""
    rng = np.random.default_rng(SEED + 99)
    texts = [d["captions_train"][int(i)]["caption"] for i in rng.choice(len(d["captions_train"]), 1000, replace=False)]
    texts += [p["question"] for p in d["presence"][:500] + d["counting"][:250] + d["vqa_train"][:250]]
    ids = torch.tensor([t for s in texts for t in bld.tok(s, add_special_tokens=False).input_ids], device=dev)
    tnorm = float(llm.get_input_embeddings()(ids).float().norm(dim=-1).mean())
    keys = [d["captions_train"][int(i)]["image"] for i in rng.choice(len(d["captions_train"]), 256, replace=False)]
    conn.scale.fill_(1.0)
    cnorm = float(conn(feats.get(keys, dev).float()).norm(dim=-1).mean())
    conn.scale.fill_(tnorm / cnorm)
    after = float(conn(feats.get(keys, dev).float()).norm(dim=-1).mean())
    return {"rule": "scale = mean||text token embedding|| (2,000 training captions/questions) / mean||connector output at init|| (256 training images)",
            "text_embed_mean_norm": tnorm, "connector_mean_norm_unscaled": cnorm, "scale": tnorm / cnorm, "connector_mean_norm_scaled": after,
            "n_text_tokens": int(ids.numel()), "n_images": len(keys)}


def make_opt(conn, llm, st):
    groups = [{"params": [p for n, p in conn.named_parameters() if p.ndim >= 2], "lr": st["lr_c"], "weight_decay": 0.01},
              {"params": [p for n, p in conn.named_parameters() if p.ndim < 2], "lr": st["lr_c"], "weight_decay": 0.0}]
    if st["lr_l"] is not None:
        groups.append({"params": L.lora_params(llm), "lr": st["lr_l"], "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, stage_lambda(st["updates"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--pool", action="store_true", help="49 visual tokens (2x2 average pooling)")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets to exercise every code path")
    a = ap.parse_args()
    global ENDS, TOTAL, DEV_AT, CKPT_EVERY
    if a.smoke:
        for s, n in zip(STAGES, [6, 4, 4]):
            s["updates"] = n
        ENDS = np.cumsum([s["updates"] for s in STAGES]).tolist()
        TOTAL, DEV_AT, CKPT_EVERY = ENDS[-1], sorted({0, 4, 8} | set(ENDS)), 3

    out = f"outputs/task2/{a.run}"
    res_dir = f"results/task2/{a.run}"
    os.makedirs(f"{out}/recovery", exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    if os.path.exists(f"{out}/DONE.json"):
        log(f"task2 {a.run} already trained")
        return
    dev = torch.device("cuda")
    seed_all(SEED)
    from task1_ft.data import get_tok
    tok = get_tok()
    feats = Features("outputs/task2/features", pool=a.pool)
    bld = Builder(tok, feats.n_vis)
    d = load_data()

    llm = load_causal_lm(LLM, torch.float16).to(dev)
    for p in llm.parameters():
        p.requires_grad_(False)
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.config.use_cache = False
    torch.manual_seed(SEED)
    conn = Connector().to(dev)  # fp32 trainable
    scaler = torch.amp.GradScaler("cuda")

    g0, meta = 0, {"stage_time_s": {}, "stage_peak_gb": {}, "mixture_counts": {}, "skipped": [], "checks": {}}
    recs = sorted(glob.glob(f"{out}/recovery/ckpt_g*.pt"), key=lambda f: int(f.split("_g")[-1][:-3]))
    st_ck = None
    if recs:
        st_ck = torch.load(recs[-1], map_location="cpu", weights_only=False)
        g0 = st_ck["g"]
        meta = st_ck["meta"]
        log(f"resuming task2 {a.run} from g={g0}")
    if g0 >= ENDS[0]:  # LoRA exists from stage 2 on
        L.inject_lora(llm, L.T2_TARGETS, LORA_R, LORA_A, LORA_DROP, SEED)
    if st_ck:
        conn.load_state_dict(st_ck["connector"])
        if "lora" in st_ck:
            L.load_raw_state(llm, st_ck["lora"])
    else:
        meta["calibration"] = calibrate(llm, conn, feats, bld, d, dev)
        log(f"calibration: {meta['calibration']}")
        jdump(meta["calibration"], f"{res_dir}/calibration.json")
    trainable_base = sum(p.numel() for n, p in llm.named_parameters() if p.requires_grad and "lora_" not in n)
    assert trainable_base == 0, "original Qwen weights must stay frozen"

    dlog = JsonlLog(f"{res_dir}/dev_log.jsonl", keep_upto=g0)
    tlog = JsonlLog(f"{res_dir}/train_log.jsonl", keep_upto=g0)
    llm.train()
    if g0 == 0:
        t0 = time.time()
        r = dev_eval(llm, conn, feats, bld, d, dev)
        dlog.write({"update": 0, "stage": "init", **r, "eval_s": time.time() - t0})
        log(f"dev @0: {r}")

    cur_stage, opt, sched, rng = None, None, None, None
    if st_ck:
        cur_stage = stage_of(g0 + 1) if g0 < TOTAL else None
        if cur_stage is not None and st_ck["stage"] == cur_stage:
            opt, sched = make_opt(conn, llm, STAGES[cur_stage])
            opt.load_state_dict(st_ck["opt"])
            sched.load_state_dict(st_ck["sched"])
            rng = np.random.default_rng()
            rng.bit_generator.state = st_ck["sampler_state"]
        else:
            cur_stage = None  # checkpoint sat exactly on a stage end: next stage starts fresh below
        scaler.load_state_dict(st_ck["scaler"])
        set_rng_state(st_ck["rng"])
    t_stage = time.time()

    def save_ckpt(g, si):
        st = {"g": g, "stage": si, "connector": conn.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "rng": rng_state(), "meta": meta, "examples_seen": g * EFF,
              "sampler_state": rng.bit_generator.state if rng is not None else None}
        if si >= 1:
            st["lora"] = L.raw_state(llm)
        f = f"{out}/recovery/ckpt_g{g}.pt"
        torch.save(st, f + ".tmp")
        os.replace(f + ".tmp", f)
        olds = sorted(glob.glob(f"{out}/recovery/ckpt_g*.pt"), key=lambda x: int(x.split("_g")[-1][:-3]))
        for o in olds[:-3]:  # keep the 3 newest + stage ends
            if int(o.split("_g")[-1][:-3]) not in ENDS:
                os.remove(o)

    for g in range(g0 + 1, TOTAL + 1):
        si = stage_of(g)
        st = STAGES[si]
        if cur_stage != si:  # new stage: (LoRA), fresh optimizer + per-stage schedule, seeded sampler
            if si == 1 and not L.lora_modules(llm):
                L.inject_lora(llm, L.T2_TARGETS, LORA_R, LORA_A, LORA_DROP, SEED)
                llm.train()
            opt, sched = make_opt(conn, llm, st)
            rng = np.random.default_rng(SEED + si) if st["mix"] else None
            cur_stage = si
            t_stage = time.time()
            torch.cuda.reset_peak_memory_stats()
            n_tr = sum(p.numel() for p in conn.parameters()) + sum(p.numel() for p in L.lora_params(llm))
            meta.setdefault("trainable_params", {})[st["name"]] = {"total": n_tr, "connector": sum(p.numel() for p in conn.parameters()),
                                                                   "lora": sum(p.numel() for p in L.lora_params(llm))}
            log(f"stage {si + 1} {st['name']}: trainable {n_tr:,}")
        counts = Counter()
        t_up = time.time()
        exs = stage1_examples(d, g) if st["mix"] is None else draw_update(d, st["mix"], rng, counts)
        for k, v in counts.items():
            meta["mixture_counts"].setdefault(st["name"], {}).setdefault(k, 0)
            meta["mixture_counts"][st["name"]][k] += v
        tot, kinds = 0.0, Counter(e["kind"] for e in exs)
        for m in range(ACCUM):
            b = bld.batch(exs[m * MICRO : (m + 1) * MICRO], dev)
            with torch.autocast("cuda", dtype=torch.float16):
                ce, _ = per_example_ce(llm, conn, feats, b, dev)
                loss = ce.sum() / EFF  # mean over tokens within example, then over the 16 examples
            scaler.scale(loss).backward()
            tot += float(loss)
        scaler.unscale_(opt)
        params = [p for grp in opt.param_groups for p in grp["params"]]
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if "connector_grad_check" not in meta["checks"] and torch.isfinite(gn):
            cg = float(torch.norm(torch.stack([p.grad.norm() for p in conn.parameters() if p.grad is not None])))
            meta["checks"]["connector_grad_check"] = {"update": g, "connector_grad_norm": cg, "ok": cg > 0}
            assert cg > 0, "no gradient reached the connector"
        lrs = [grp["lr"] for grp in opt.param_groups]
        s0 = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        sk = scaler.get_scale() < s0
        if sk:
            meta["skipped"].append(g)
        sched.step()
        opt.zero_grad(set_to_none=True)
        dt = time.time() - t_up
        meta["stage_time_s"][st["name"]] = meta["stage_time_s"].get(st["name"], 0.0) + dt
        meta["stage_peak_gb"][st["name"]] = max(meta["stage_peak_gb"].get(st["name"], 0.0), gpu_mem()["peak_alloc_gb"])
        tlog.write({"update": g, "stage": st["name"], "loss": tot, "lr_connector": lrs[0],
                    "lr_lora": lrs[2] if len(lrs) > 2 else None, "grad_norm": float(gn),
                    "scale": s0, "skipped": bool(sk), "kinds": dict(kinds), "draws": dict(counts), "step_s": dt})
        if g % 25 == 0:
            log(f"task2 {a.run} g={g}/{TOTAL} [{st['name']}] loss={tot:.4f} gn={float(gn):.2f} {dt:.2f}s/update mem={gpu_mem()['peak_alloc_gb']:.1f}GB")
        if g in DEV_AT:
            t0 = time.time()
            r = dev_eval(llm, conn, feats, bld, d, dev)
            dlog.write({"update": g, "stage": st["name"], **r, "eval_s": time.time() - t0})
            log(f"dev @{g}: caption {r['caption_loss']:.3f} | grounding swap {r['grounding_dev']['swap_penalty']:.3f} | vqa swap {r['vqa_dev']['swap_penalty']:.3f}")
        if g in ENDS:
            sd = f"{out}/stage{si + 1}"
            os.makedirs(sd, exist_ok=True)
            save_file({k: v.contiguous() for k, v in conn.state_dict().items()}, f"{sd}/connector.safetensors")
            if L.lora_modules(llm):
                save_file(L.raw_state(llm), f"{sd}/lora.safetensors")
            jdump({"stage": st["name"], "g": g, "bytes": path_size(sd)}, f"{sd}/info.json")
        if g % CKPT_EVERY == 0 or g in ENDS:
            save_ckpt(g, si)
        if g < TOTAL and past_deadline(margin=600):
            save_ckpt(g, si)
            log(f"task2 {a.run}: deadline, stopping at g={g}")
            sys.exit(EXIT_PREEMPT)

    meta.update({"run": a.run, "n_visual_tokens": feats.n_vis, "total_updates": TOTAL, "examples": TOTAL * EFF, "stages": STAGES,
                 "question_truncations": bld.n_q_trunc, "train_time_s": sum(meta["stage_time_s"].values()), "env": env_info(),
                 "lora": {"r": LORA_R, "alpha": LORA_A, "dropout": LORA_DROP, "targets": L.T2_TARGETS},
                 "optimizer": "AdamW wd 0.01 (no decay on biases), fresh optimizer per stage, per-stage 5% warmup + cosine to 10% peak",
                 "checkpoint_bytes": path_size(f"{out}/recovery")})
    jdump(meta, f"{res_dir}/train_summary.json")
    jdump(meta, f"{out}/DONE.json")
    log(f"task2 {a.run} done: {meta['train_time_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
