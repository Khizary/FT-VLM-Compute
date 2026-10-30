"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16 on 40k MetaMathQA examples.

python -m task1_ft.train --run fullft|l2sp|lora4|lora16|oplora
Each run starts from the original Qwen3-0.6B, sees the same examples in the same order, one epoch,
1,250 updates of 32 examples. Resumes automatically from the newest recovery checkpoint it finds.
"""
import argparse
import glob
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from common import lora as L
from common.utils import (SEED, JsonlLog, env_info, gpu_mem, jdump, load_causal_lm, log, past_deadline, path_size,
                          rng_state, seed_all, set_rng_state)
from task1_ft.data import MODEL, get_tok, orig_weights

RUNS = {
    "fullft": dict(method="full", lr=2e-5, lam=0.0),
    "l2sp": dict(method="full", lr=2e-5, lam=1e-3),
    "lora4": dict(method="lora", lr=2e-4, r=4, alpha=8),
    "lora16": dict(method="lora", lr=2e-4, r=16, alpha=32),
    "oplora": dict(method="oplora", lr=2e-4, r=16, alpha=32, k=128),
}
TOTAL = 1250
WARMUP = 38
MICRO, ACCUM = 2, 16
EFF = MICRO * ACCUM
CKPT_AT = [312, 625, 937, 1250]
MID = 625
DIAG_LAYERS = [0, 13, 27]
CE_CHUNK = 1024
LORA_DROPOUT = 0.05
EXIT_PREEMPT = 75


def lr_lambda(i):
    # i = scheduler steps taken so far, so update u = i + 1. linear warmup over 38 updates, cosine to 0 at 1250
    u = i + 1
    if u <= WARMUP:
        return u / WARMUP
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (u - WARMUP) / (TOTAL - WARMUP))))


def make_micro(data, idxs, pad_id, dev):
    seqs, labs = [], []
    for i in idxs:
        p, r = data["prompt_ids"][i], data["resp_ids"][i]
        seqs.append(p + r)
        labs.append([-100] * len(p) + r)  # response tokens + <|im_end|> are supervised
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), T), dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    for j, (s, l) in enumerate(zip(seqs, labs)):
        ids[j, : len(s)] = torch.tensor(s)
        am[j, : len(s)] = 1
        lab[j, : len(l)] = torch.tensor(l)
    return ids.to(dev), am.to(dev), lab.to(dev)


def _ce_chunk(head, h, y):
    return F.cross_entropy(head(h).float(), y, reduction="sum")


def loss_sum(model, ids, am, lab):
    """summed next-token CE over supervised positions; logits only built for those positions, in chunks"""
    h = model.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    h, y = h[:, :-1], lab[:, 1:]
    sel = y != -100
    hs, ys = h[sel], y[sel]
    tot = 0.0
    for c in range(0, hs.shape[0], CE_CHUNK):
        tot = tot + checkpoint(_ce_chunk, model.lm_head, hs[c : c + CE_CHUNK], ys[c : c + CE_CHUNK], use_reentrant=False)
    return tot


def param_groups(params_named, wd=0.01):
    decay = [p for n, p in params_named if p.ndim >= 2 and not n.endswith(".bias")]
    nod = [p for n, p in params_named if not (p.ndim >= 2 and not n.endswith(".bias"))]
    groups = [{"params": decay, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


@torch.no_grad()
def displacement(named, ref):
    tot = torch.zeros((), dtype=torch.float64, device=named[0][1].device)
    for n, p in named:
        tot += (p.data.float() - ref[n].to(p.device).float()).pow(2).sum(dtype=torch.float64)
    return float(tot.sqrt())


@torch.no_grad()
def l2sp_apply(named, ref_gpu, lam):
    """adds grad of lam*Omega = lam*(theta-theta0) to the (unscaled) grads; returns Omega = 0.5*sum (theta-theta0)^2"""
    om = torch.zeros((), dtype=torch.float64, device=named[0][1].device)
    for n, p in named:
        d = p.data - ref_gpu[n].float()
        om += d.pow(2).sum(dtype=torch.float64)
        if p.grad is not None:
            p.grad.add_(d, alpha=lam)
        del d
    return 0.5 * float(om)


def find_recovery(dirs):
    best = None
    for d in dirs:
        for f in glob.glob(os.path.join(d, "recovery_u*.pt")):
            u = int(os.path.basename(f)[len("recovery_u") : -3])
            if best is None or u > best[0]:
                best = (u, f)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--out-root", default="outputs/task1")
    ap.add_argument("--scratch", default=os.environ.get("PA3_SCRATCH", "/tmp/pa3_scratch"))
    ap.add_argument("--max-updates", type=int, default=TOTAL)
    ap.add_argument("--ckpt-at", default=",".join(map(str, CKPT_AT)))
    ap.add_argument("--mid", type=int, default=MID)
    ap.add_argument("--stop-after", type=int, default=None, help="simulate a preemption after this update (smoke test)")
    a = ap.parse_args()

    cfg = RUNS[a.run]
    full = cfg["method"] == "full"
    final_u = a.max_updates
    ckpt_at = sorted({int(x) for x in a.ckpt_at.split(",") if int(x) <= final_u} | {final_u})
    out = os.path.join(a.out_root, a.run)
    rec_dirs = [os.path.join(out, "recovery"), os.path.join(a.scratch, "task1", a.run)]
    # adapter checkpoints are small -> keep all in the saved output; full-model ones (~7 GB with Adam) go to scratch
    rec_dir = rec_dirs[1] if full else rec_dirs[0]
    os.makedirs(out, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    if os.path.exists(os.path.join(out, "DONE.json")):
        log(f"{a.run} already done")
        return
    dev = torch.device("cuda")
    seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    t_wall0 = time.time()

    tok = get_tok()
    data = torch.load(os.path.join(a.out_root, "data", "train_tok.pt"))
    assert len(data["resp_ids"]) >= final_u * EFF
    sd0 = orig_weights()  # bf16 hub tensors = theta0 exactly
    overhead = {}

    if full:
        model = load_causal_lm(MODEL, torch.float32).to(dev)  # fp32 master weights
        named = [(n, p) for n, p in model.named_parameters()]  # unique params (tied embedding counted once)
        for n, p in named:
            p.requires_grad_(True)
        ref_gpu = None
        if cfg["lam"] > 0:
            torch.cuda.synchronize()
            t0 = time.time()
            ref_gpu = {n: sd0[n].to(dev, torch.bfloat16) for n, _ in named}
            torch.cuda.synchronize()
            overhead["l2sp_ref_transfer_s"] = time.time() - t0
            overhead["l2sp_ref_bytes"] = sum(t.numel() * t.element_size() for t in ref_gpu.values())
            overhead["l2sp_ref_dtype"] = "bfloat16 (lossless: the hub checkpoint is bf16)"
            mism = max(float((p.data - ref_gpu[n].float()).abs().max()) for n, p in named)
            assert mism == 0.0, f"theta0 copy not exact: {mism}"
            overhead["l2sp_penalty_time_s"] = 0.0
    else:
        model = load_causal_lm(MODEL, torch.float16).to(dev)  # frozen fp16 backbone
        bases = None
        if cfg["method"] == "oplora":
            names = [n for n, _ in L.lora_targets(model, L.T1_TARGETS)]
            torch.cuda.synchronize()
            t0 = time.time()
            bases = L.compute_bases(sd0, names, cfg["k"], dev)
            torch.cuda.synchronize()
            overhead["oplora_svd_time_s"] = time.time() - t0
            overhead["oplora_basis_bytes"] = sum((U.numel() + V.numel()) * 4 for U, V in bases.values())
            overhead["oplora_k"] = cfg["k"]
        L.inject_lora(model, L.T1_TARGETS, cfg["r"], cfg["alpha"], LORA_DROPOUT, SEED, bases)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        overhead["init_A_checksum"] = float(sum(float(m.lora_A.double().sum()) for m in L.lora_modules(model).values()))
        overhead["n_adapted_modules"] = len(L.lora_modules(model))
    params = [p for _, p in named]
    n_trainable = sum(p.numel() for p in params)
    log(f"{a.run}: trainable params {n_trainable:,}  overhead {overhead}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(param_groups(named), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    start_u, train_time, skipped, prev_peak, sessions = 0, 0.0, [], 0.0, 0
    rec = find_recovery(rec_dirs)
    if rec:
        log(f"resuming {a.run} from {rec[1]}")
        st = torch.load(rec[1], map_location="cpu", weights_only=False)
        if full:
            model.load_state_dict(st["trainable"])
        else:
            L.load_raw_state(model, st["trainable"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start_u, train_time, skipped = st["update"], st["train_time"], st["skipped"]
        prev_peak, sessions = st.get("peak_alloc_gb", 0.0), st.get("sessions", 0)
        overhead.update(st.get("overhead", {}))
        del st
    sessions += 1
    tlog = JsonlLog(os.path.join(out, "train_log.jsonl"), keep_upto=start_u)
    pad_id = tok.pad_token_id

    def save_recovery(u, reason):
        t0 = time.time()
        d = rec_dir if reason == "scheduled" else rec_dirs[0]  # deadline checkpoints always land in saved output
        os.makedirs(d, exist_ok=True)
        st = {
            "update": u, "examples_seen": u * EFF, "run": a.run, "reason": reason,
            "trainable": model.state_dict() if full else L.raw_state(model),
            "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
            "train_time": train_time, "skipped": skipped, "overhead": overhead, "sessions": sessions,
            "peak_alloc_gb": max(prev_peak, gpu_mem()["peak_alloc_gb"]), "data_position": u * EFF,
        }
        f = os.path.join(d, f"recovery_u{u}.pt")
        torch.save(st, f + ".tmp")
        os.replace(f + ".tmp", f)
        size = os.path.getsize(f)
        if full or reason != "scheduled":  # keep only the newest full-model checkpoint (disk)
            for dd in rec_dirs:
                for g in glob.glob(os.path.join(dd, "recovery_u*.pt")):
                    if g != f and (full or "deadline" in torch.load(g, map_location="cpu", weights_only=False).get("reason", "")):
                        os.remove(g)
        log(f"recovery checkpoint u={u} ({reason}) {size / 2**30:.2f} GB in {time.time() - t0:.0f}s")
        return size

    def export(tag, u):
        t0 = time.time()
        d = os.path.join(out, tag)
        os.makedirs(d, exist_ok=True)
        info = {"tag": tag, "update": u, "examples_seen": u * EFF}
        with torch.no_grad():
            if full:
                tied = getattr(model.config, "tie_word_embeddings", False)
                sd = {k: v.detach().to(torch.float16).cpu().contiguous() for k, v in model.state_dict().items() if not (tied and k == "lm_head.weight")}
                save_file(sd, os.path.join(d, "model_fp16.safetensors"))
                diag = {}
                for li in DIAG_LAYERS:
                    k = f"model.layers.{li}.mlp.down_proj.weight"
                    diag[k] = model.get_parameter(k).detach().float().cpu().contiguous()
                save_file(diag, os.path.join(d, "down_proj_fp32.safetensors"))
                info["displacement_l2"] = displacement(named, sd0)
                del sd
            else:
                save_file(L.effective_state(model), os.path.join(d, "adapter.safetensors"))  # projected for OPLoRA
                save_file(L.raw_state(model), os.path.join(d, "adapter_raw.safetensors"))
                info.update(r=cfg["r"], alpha=cfg["alpha"], scaling=cfg["alpha"] / cfg["r"], targets=L.T1_TARGETS, method=cfg["method"])
                if cfg["method"] == "oplora":
                    worst_b = worst_a = 0.0
                    for m in L.lora_modules(model).values():
                        A, B = m.factors()
                        worst_b = max(worst_b, float((m.Uk.T @ B).norm() / (B.norm() + 1e-12)))
                        worst_a = max(worst_a, float((A @ m.Vk).norm() / (A.norm() + 1e-12)))
                    info["constraint_max_rel_UkT_B"] = worst_b
                    info["constraint_max_rel_A_Vk"] = worst_a
        info["export_time_s"] = time.time() - t0
        info["export_bytes"] = path_size(d)
        jdump(info, os.path.join(d, "export_info.json"))
        log(f"exported {tag}: {info}")
        return info

    t_last = time.time()
    tok_since = 0
    for u in range(start_u + 1, final_u + 1):
        t_up = time.time()
        idx = list(range((u - 1) * EFF, u * EFF))
        n_valid = sum(len(data["resp_ids"][i]) for i in idx)  # loss normalised over all valid tokens in the effective batch
        sft = 0.0
        for m in range(ACCUM):
            ids, am, lab = make_micro(data, idx[m * MICRO : (m + 1) * MICRO], pad_id, dev)
            with torch.autocast("cuda", dtype=torch.float16):
                ls = loss_sum(model, ids, am, lab)
            scaler.scale(ls / n_valid).backward()
            sft += float(ls)
            tok_since += int(am.sum())
        scaler.unscale_(opt)
        row = {"update": u, "examples_seen": u * EFF, "lr": opt.param_groups[0]["lr"], "sft_loss": sft / n_valid, "n_tokens": n_valid}
        if full and cfg["lam"] > 0:
            t0 = time.time()
            om = l2sp_apply(named, ref_gpu, cfg["lam"])
            overhead["l2sp_penalty_time_s"] += time.time() - t0
            row.update(omega=om, lam_omega=cfg["lam"] * om, total_loss=row["sft_loss"] + cfg["lam"] * om)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        s0 = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        sk = scaler.get_scale() < s0
        if sk:
            skipped.append(u)
            log(f"update {u} skipped (non-finite grads), scale {s0} -> {scaler.get_scale()}")
        sched.step()
        opt.zero_grad(set_to_none=True)
        train_time += time.time() - t_up
        row.update(grad_norm=float(gn), scale=s0, skipped=bool(sk), train_time_s=train_time)
        tlog.write(row)

        if u % 10 == 0 or u == start_u + 1:
            dt = time.time() - t_last
            log(f"{a.run} u={u}/{final_u} loss={row['sft_loss']:.4f} lr={row['lr']:.2e} gn={float(gn):.3f} "
                f"tok/s={tok_since / max(dt, 1e-6):.0f} eta={(final_u - u) * (time.time() - t_wall0) / max(1, u - start_u) / 3600:.2f}h "
                f"mem={gpu_mem().get('peak_alloc_gb', 0):.1f}GB")
            t_last, tok_since = time.time(), 0

        if u == a.mid:
            export("mid", u)
        if u == final_u:
            export("final", u)
        if u in ckpt_at:
            overhead.setdefault("recovery_ckpt_bytes", {})[u] = save_recovery(u, "scheduled")
        if u < final_u and (past_deadline(margin=600) or (a.stop_after and u == a.stop_after)):
            save_recovery(u, "deadline")
            log(f"{a.run}: stopping at update {u} for deadline/preemption; rerun to resume")
            sys.exit(EXIT_PREEMPT)

    mem = gpu_mem()
    summary = {
        "run": a.run, "config": cfg, "seed": SEED, "updates": final_u, "examples": final_u * EFF,
        "micro_batch": MICRO, "grad_accum": ACCUM, "warmup": WARMUP, "schedule": "linear warmup 38, cosine to 0 at 1250",
        "precision": "fp16 autocast + dynamic GradScaler; fp32 master (full) / fp16 frozen backbone + fp32 adapters",
        "trainable_params": n_trainable, "train_time_s": train_time, "train_time_h": train_time / 3600,
        "wall_time_this_session_s": time.time() - t_wall0, "sessions": sessions,
        "peak_alloc_gb": max(prev_peak, mem["peak_alloc_gb"]), "peak_reserved_gb_last_session": mem["peak_reserved_gb"],
        "skipped_updates": skipped, "overhead": overhead, "env": env_info(),
        "exports": {t: os.path.join(out, t, "export_info.json") for t in ["mid", "final"] if os.path.exists(os.path.join(out, t, "export_info.json"))},
    }
    jdump(summary, os.path.join("results/task1", f"train_{a.run}.json"))
    jdump(summary, os.path.join(out, "DONE.json"))
    shutil.copy(os.path.join(out, "train_log.jsonl"), os.path.join("results/task1", f"train_log_{a.run}.jsonl"))
    log(f"{a.run} finished: {train_time / 3600:.2f} h, peak {summary['peak_alloc_gb']:.1f} GB, skipped {len(skipped)}")


if __name__ == "__main__":
    main()
