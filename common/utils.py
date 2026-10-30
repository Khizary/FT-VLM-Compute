import json
import os
import random
import time

import numpy as np
import torch

SEED = 6304


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl_write(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def jsonl_read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class JsonlLog:
    """append-only jsonl, flushed every line so a killed session keeps its logs"""

    def __init__(self, path, keep_upto=None, key="update"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # on resume drop entries written after the checkpoint we resumed from
        if keep_upto is not None and os.path.exists(path):
            rows = [r for r in jsonl_read(path) if r.get(key, -1) <= keep_upto]
            jsonl_write(rows, path)
        self.f = open(path, "a")

    def write(self, d):
        self.f.write(json.dumps(d, default=str) + "\n")
        self.f.flush()


def rng_state():
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def gpu_mem():
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }


def path_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
    return tot


def deadline():
    """unix time after which long jobs should checkpoint and stop (set by the kaggle runner)"""
    d = os.environ.get("PA3_DEADLINE")
    return float(d) if d else None


def past_deadline(margin=0.0):
    d = deadline()
    return d is not None and time.time() + margin > d


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def autocast_on():
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:
        return torch.is_autocast_enabled()


def load_causal_lm(name, dtype, revision=None):
    """dtype kwarg got renamed in transformers 4.56; old versions silently ignore `dtype`"""
    import transformers
    from transformers import AutoModelForCausalLM

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
    if revision:
        kw["revision"] = revision
    m = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa", **kw)
    assert next(m.parameters()).dtype == dtype, f"model loaded as {next(m.parameters()).dtype}, wanted {dtype}"
    return m


def chat_prompt(tok, messages, **kw):
    """generation prompt with thinking disabled (Qwen3 inserts an empty <think></think> block)"""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def env_info():
    import platform

    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    for mod in ["lm_eval", "datasets", "math_verify", "safetensors"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info
