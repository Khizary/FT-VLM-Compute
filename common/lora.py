"""Plain LoRA / OPLoRA linear wrapper (written by hand so OPLoRA's projection is explicit).

W = W0 + (alpha/r) B A,  A in R^{r x din} (random), B in R^{dout x r} (zeros).
OPLoRA uses effective factors  B~ = B - Uk (Uk^T B),  A~ = A - (A Vk) Vk^T
where Uk, Vk are the leading-k singular bases of the *starting* weight W0 (fp32, frozen).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import autocast_on

T1_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
T2_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout, A_init, Uk=None, Vk=None):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(A_init.to(dev, torch.float32).clone())
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.protected = Uk is not None
        if self.protected:
            self.register_buffer("Uk", Uk.to(dev, torch.float32), persistent=False)
            self.register_buffer("Vk", Vk.to(dev, torch.float32), persistent=False)

    def factors(self):
        """effective (A, B); projections done in fp32 even inside an autocast region"""
        A, B = self.lora_A, self.lora_B
        if self.protected:
            with torch.autocast(device_type="cuda", enabled=False):
                B = B - self.Uk @ (self.Uk.T @ B)
                A = A - (A @ self.Vk) @ self.Vk.T
        return A, B

    def delta_weight(self):
        A, B = self.factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        out = self.base(x)
        A, B = self.factors()
        h = self.drop(x)
        if not autocast_on():
            h = h.to(A.dtype)
        lora = F.linear(F.linear(h, A), B) * self.scaling
        return out + lora.to(out.dtype)


def lora_targets(model, targets):
    """(name, module) for every targeted nn.Linear, in the model's deterministic module order"""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            out.append((name, mod))
    return out


def init_A(shape, gen):
    # same as PEFT / nn.Linear default: kaiming_uniform(a=sqrt(5)) -> U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    bound = 1.0 / math.sqrt(shape[1])
    return (torch.rand(shape, generator=gen) * 2 - 1) * bound


def inject_lora(model, targets, r, alpha, dropout, seed, bases=None):
    """Replace target linears with LoRALinear. A factors come from a dedicated CPU generator seeded
    with `seed`, so LoRA-16 and OPLoRA-16 start from identical unprojected factors."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    wrapped = {}
    for name, mod in lora_targets(model, targets):
        A0 = init_A((r, mod.in_features), gen)
        Uk = Vk = None
        if bases is not None:
            Uk, Vk = bases[name]
        new = LoRALinear(mod, r, alpha, dropout, A0, Uk, Vk)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], new)
        wrapped[name] = new
    return wrapped


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_params(model):
    ps = []
    for m in lora_modules(model).values():
        ps += [m.lora_A, m.lora_B]
    return ps


def raw_state(model):
    out = {}
    for n, m in lora_modules(model).items():
        out[n + ".lora_A"] = m.lora_A.detach().float().cpu().contiguous()
        out[n + ".lora_B"] = m.lora_B.detach().float().cpu().contiguous()
    return out


def load_raw_state(model, sd):
    for n, m in lora_modules(model).items():
        m.lora_A.data.copy_(sd[n + ".lora_A"])
        m.lora_B.data.copy_(sd[n + ".lora_B"])


@torch.no_grad()
def effective_state(model):
    """projected factors for OPLoRA (== raw factors for plain LoRA); this is what gets exported/merged"""
    out = {}
    for n, m in lora_modules(model).items():
        A, B = m.factors()
        out[n + ".lora_A"] = A.detach().float().cpu().contiguous()
        out[n + ".lora_B"] = B.detach().float().cpu().contiguous()
    return out


@torch.no_grad()
def compute_bases(orig_weights, names, k, device):
    """leading-k left/right singular bases of each fp32 starting weight"""
    bases = {}
    for n in names:
        W = orig_weights[n + ".weight"].to(device, torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        bases[n] = (U[:, :k].contiguous(), Vh[:k].T.contiguous())
        del U, S, Vh, W
    return bases


@torch.no_grad()
def merge_factors(model, factors, scaling):
    """W <- W0 + scaling * B A on a fresh backbone; computed in fp32 then cast back to the model dtype"""
    names = sorted({k.rsplit(".", 1)[0] for k in factors})
    for n in names:
        lin = model.get_submodule(n)
        A = factors[n + ".lora_A"].to(lin.weight.device, torch.float32)
        B = factors[n + ".lora_B"].to(lin.weight.device, torch.float32)
        lin.weight.data = (lin.weight.data.float() + scaling * (B @ A)).to(lin.weight.dtype)
    return len(names)


def attach_fixed_factors(model, factors, r, alpha):
    """unmerged adapter model built from exported (already projected) factors; used to check merging"""
    gen = torch.Generator().manual_seed(0)
    names = sorted({k.rsplit(".", 1)[0] for k in factors})
    for n in names:
        mod = model.get_submodule(n)
        new = LoRALinear(mod, r, alpha, 0.0, init_A((r, mod.in_features), gen))
        new.lora_A.data.copy_(factors[n + ".lora_A"])
        new.lora_B.data.copy_(factors[n + ".lora_B"])
        parent = model.get_submodule(n.rsplit(".", 1)[0])
        setattr(parent, n.rsplit(".", 1)[1], new)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
