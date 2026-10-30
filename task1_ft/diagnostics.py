"""Task 1 weight-space + activation diagnostics on down_proj of layers 0, 13, 27 (fp32).

rho_k = ||Uk Uk^T dW||_F^2 / ||dW||_F^2,  eta = ||dW||_F / ||W0||_F,  R(z) = ||dW z|| / (||W0 z|| + 1e-8)
z = starting model's down_proj input at the last prompt token of each probe (32 GSM8K + 32 HellaSwag).
python -m task1_ft.diagnostics
"""
import argparse
import os

import numpy as np
import torch
from safetensors.torch import load_file

from common.utils import jdump, jload, load_causal_lm, log
from task1_ft.data import MODEL, get_tok, orig_weights, probe_prompts

RES = "results/task1"
LAYERS = [0, 13, 27]
K = 128
CKPTS = {"fullft_mid": ("full", "fullft/mid"), "fullft": ("full", "fullft/final"), "l2sp": ("full", "l2sp/final"),
         "lora4": ("adapter", "lora4/final"), "lora16": ("adapter", "lora16/final"), "oplora": ("adapter", "oplora/final")}


def key(li):
    return f"model.layers.{li}.mlp.down_proj"


@torch.no_grad()
def cache_inputs(dev, tok, probes):
    model = load_causal_lm(MODEL, torch.float32).to(dev).eval()
    store = {li: [] for li in LAYERS}
    hooks = []
    cur = {}
    for li in LAYERS:
        def hook(mod, inp, li=li):
            cur[li] = inp[0][0, -1].detach().float().cpu()  # single unpadded prompt -> last token is the last prompt token
        hooks.append(model.get_submodule(key(li)).register_forward_pre_hook(hook))
    for text in probe_prompts(tok, probes):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        model(ids)
        for li in LAYERS:
            store[li].append(cur[li])
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {li: torch.stack(v) for li, v in store.items()}  # (64, 3072)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/task1")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tok()
    probes = jload(f"{RES}/probe_set.json")["probes"]
    is_math = np.array([p["set"] == "math" for p in probes])
    sd0 = orig_weights()
    W0 = {li: sd0[key(li) + ".weight"].float().to(dev) for li in LAYERS}
    Uk = {}
    for li in LAYERS:
        U, S, Vh = torch.linalg.svd(W0[li], full_matrices=False)
        Uk[li] = U[:, :K]
    Z = {li: z.to(dev) for li, z in cache_inputs(dev, tok, probes).items()}

    out = {"layers": LAYERS, "k": K, "probe_ids": [(p["source"], p["idx"]) for p in probes], "checkpoints": {}}
    for name, (kind, rel) in CKPTS.items():
        d = os.path.join(a.out_root, rel)
        if not os.path.exists(d):
            log(f"skip {name}: {d} missing")
            continue
        if kind == "full":
            tr = load_file(os.path.join(d, "down_proj_fp32.safetensors"))
            dW = {li: tr[key(li) + ".weight"].to(dev).float() - W0[li] for li in LAYERS}
        else:
            ex = jload(os.path.join(d, "export_info.json"))
            fac = load_file(os.path.join(d, "adapter.safetensors"))  # projected factors for OPLoRA
            dW = {li: ex["scaling"] * fac[key(li) + ".lora_B"].to(dev) @ fac[key(li) + ".lora_A"].to(dev) for li in LAYERS}
        res = {}
        for li in LAYERS:
            D = dW[li]
            nD = float(D.norm())
            proj = Uk[li] @ (Uk[li].T @ D)
            rho = float(proj.norm() ** 2 / (nD ** 2)) if nD > 0 else None
            eta = nD / float(W0[li].norm())
            R = ((D @ Z[li].T).norm(dim=0) / ((W0[li] @ Z[li].T).norm(dim=0) + 1e-8)).cpu().numpy()
            res[str(li)] = {
                "rho_k": rho, "eta": eta, "dW_fro": nD, "W0_fro": float(W0[li].norm()),
                "R_math_mean": float(R[is_math].mean()), "R_math_median": float(np.median(R[is_math])),
                "R_nonmath_mean": float(R[~is_math].mean()), "R_nonmath_median": float(np.median(R[~is_math])),
                "R_math_over_nonmath": float(R[is_math].mean() / max(R[~is_math].mean(), 1e-12)),
                "R_per_probe": R.tolist(),
            }
        out["checkpoints"][name] = res
        log(f"{name}: " + ", ".join(f"L{li} rho={res[str(li)]['rho_k']:.4g} eta={res[str(li)]['eta']:.3g} "
                                    f"Rm={res[str(li)]['R_math_mean']:.3g} Rn={res[str(li)]['R_nonmath_mean']:.3g}" for li in LAYERS))
    jdump(out, f"{RES}/diagnostics.json")


if __name__ == "__main__":
    main()
