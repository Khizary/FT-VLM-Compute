"""Cache frozen SigLIP patch features (196 x 768, final post-layernorm applied once) in fp16.

python -m task2_vlm.cache_features
Images come from images.cocodataset.org (originals); falls back to any mounted Kaggle COCO-2014 copy.
"""
import glob
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from common.utils import SEED, gpu_mem, jdump, jload, log, seed_all

ENC = "google/siglip-base-patch16-224"
DATA = "results/task2/data"
FEAT_DIR = "outputs/task2/features"
IMG_DIR = os.environ.get("PA3_IMG", "/tmp/coco_img")


def fname(key):
    split, iid = key.split(":")
    return f"{split}2014", f"COCO_{split}2014_{int(iid):012d}.jpg"


def mounted_dirs():
    return {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/train2014", recursive=True) + glob.glob("/kaggle/input/**/val2014", recursive=True)}


def fetch(key, mounts):
    sub, fn = fname(key)
    dst = os.path.join(IMG_DIR, sub, fn)
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(f"http://images.cocodataset.org/{sub}/{fn}", dst + ".part")
            os.replace(dst + ".part", dst)
            return dst
        except Exception:
            time.sleep(1 + attempt)
    if sub in mounts and os.path.exists(os.path.join(mounts[sub], fn)):
        return os.path.join(mounts[sub], fn)
    raise RuntimeError(f"could not get {key}")


def needed_images():
    keys = set()
    for c in jload(f"{DATA}/captions_train.json") + jload(f"{DATA}/captions_dev.json"):
        keys.add(c["image"])
    for f in ["grounding_train_pool", "grounding_dev", "grounding_test", "vqa_train", "vqa_dev", "vqa_test"]:
        for p in jload(f"{DATA}/{f}.json"):
            for s in p["sides"]:
                keys.add(s["image"])
    return sorted(keys)


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


def main():
    from transformers import SiglipImageProcessor, SiglipVisionModel

    if os.path.exists(f"{FEAT_DIR}/index.json"):
        log("features already cached")
        return
    seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    keys = needed_images()
    log(f"{len(keys)} images to cache")
    t0 = time.time()
    mounts = mounted_dirs()
    with ThreadPoolExecutor(32) as ex:
        paths = list(ex.map(lambda k: fetch(k, mounts), keys))
    t_dl = time.time() - t0
    log(f"images ready in {t_dl:.0f}s")

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    proc = SiglipImageProcessor.from_pretrained(ENC)  # deterministic: resize 224 bicubic, rescale, normalise
    enc = SiglipVisionModel.from_pretrained(ENC).float().to(dev).eval()
    os.makedirs(FEAT_DIR, exist_ok=True)
    arr = np.lib.format.open_memmap(f"{FEAT_DIR}/feats.npy.tmp", mode="w+", dtype=np.float16, shape=(len(keys), 196, 768))
    dl = torch.utils.data.DataLoader(DS(paths, proc), batch_size=64, num_workers=4, shuffle=False)
    t1 = time.time()
    row, check = 0, None
    with torch.no_grad():
        for px in dl:
            out = enc(pixel_values=px.to(dev), output_hidden_states=check is None)
            h = out.last_hidden_state
            assert h.shape[1:] == (196, 768), h.shape
            if check is None:  # decide once whether last_hidden_state already includes the final post-layernorm
                post_ln = next(m for n, m in enc.named_modules() if n.endswith("post_layernorm"))  # attribute path differs across versions
                pre = out.hidden_states[-1]
                d_post = float((post_ln(pre) - h).abs().max())
                d_pre = float((pre - h).abs().max())
                apply_ln = d_post > 1e-4 and d_pre < 1e-5  # HF returned pre-LN features -> apply it ourselves, once
                assert d_post < 1e-4 or apply_ln, (d_post, d_pre)
                check = {"diff_vs_postln_of_last_encoder_state": d_post, "diff_vs_raw_last_encoder_state": d_pre,
                         "post_ln_applied_by": "script" if apply_ln else "transformers last_hidden_state", "shape": list(h.shape[1:])}
                log(f"layernorm check: {check}")
            if apply_ln:
                h = post_ln(h)
            arr[row : row + len(h)] = h.half().cpu().numpy()
            row += len(h)
    torch.cuda.synchronize()
    t_enc = time.time() - t1
    arr.flush()
    del arr
    os.replace(f"{FEAT_DIR}/feats.npy.tmp", f"{FEAT_DIR}/feats.npy")
    with open(f"{FEAT_DIR}/index.json", "w") as f:
        json.dump({k: i for i, k in enumerate(keys)}, f)
    res = {"encoder": ENC, "n_images": len(keys), "feature_shape": [196, 768], "dtype": "float16 (encoder run in fp32)", "ln_check": check,
           "image_fetch_s": t_dl, "encode_s": t_enc, "total_s": time.time() - t0, "bytes": os.path.getsize(f"{FEAT_DIR}/feats.npy"),
           "gpu": gpu_mem(), "preprocessing": proc.to_dict()}
    jdump(res, "results/task2/feature_cache.json")
    log(f"cached {len(keys)} images: {res['encode_s']:.0f}s encode, peak {res['gpu']}")


if __name__ == "__main__":
    main()
