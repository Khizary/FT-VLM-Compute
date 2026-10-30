"""Task 2 data selection (CPU). Downloads COCO-2014 captions/instances and VQAv2 jsons, then selects:
  captions: 6,000 train2014 images x 3 captions; 96 val2014 images x 1 caption (dev)
  grounding (generated from instance annotations): reusable train pools; dev 24 pairs; test 64 pairs
  VQAv2 complementary pairs (>=8/10 agreement on each side, different consensus answers):
      train 640/256/640, dev 12/8/12, test 36/24/36 (yes-no / number / other)
All eval images are val2014, unique across every eval set, and never used in training. Seed 6304.
python -m task2_vlm.prepare_data
"""
import csv
import json
import os
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np

from common.utils import SEED, jdump, log
from task2_vlm.vqa_norm import normalize

RAW = os.environ.get("PA3_RAW", "/tmp/pa3_raw")
OUT = "results/task2/data"
URLS = {
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "v2_Questions_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "v2_Questions_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "v2_Annotations_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "v2_Annotations_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "v2_Complementary_Pairs_Train_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "v2_Complementary_Pairs_Val_mscoco.zip": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
}
VQA_QUOTA = {"train": {"yes/no": 640, "number": 256, "other": 640}, "dev": {"yes/no": 12, "number": 8, "other": 12}, "test": {"yes/no": 36, "number": 24, "other": 36}}
GROUND_EVAL = {"dev": {"presence": 12, "counting": 12}, "test": {"presence": 32, "counting": 32}}
POOL_SIZE = {"presence": 4000, "counting": 2000}
PRESENT_MIN_AREA = 0.01  # positive needs a non-crowd instance covering >=1% of the image
COUNT_MIN_AREA = 0.005   # counting needs every instance >=0.5% of the image and no crowd boxes
PLURAL = {"person": "people", "bus": "buses", "sheep": "sheep", "skis": "skis", "scissors": "scissors", "knife": "knives",
          "mouse": "mice", "broccoli": "broccoli", "couch": "couches", "sandwich": "sandwiches", "bench": "benches",
          "wine glass": "wine glasses", "toothbrush": "toothbrushes", "hair drier": "hair driers", "glass": "glasses"}
NO_ARTICLE = {"skis", "scissors", "broccoli"}


def fetch_all():
    os.makedirs(RAW, exist_ok=True)
    for fn, url in URLS.items():
        p = os.path.join(RAW, fn)
        if not os.path.exists(p):
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, p + ".part")
            os.replace(p + ".part", p)
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if n.endswith(".json") and not os.path.exists(os.path.join(RAW, os.path.basename(n))):
                    with z.open(n) as src, open(os.path.join(RAW, os.path.basename(n)), "wb") as dst:
                        dst.write(src.read())


def J(name):
    with open(os.path.join(RAW, name)) as f:
        return json.load(f)


def phrase(name):
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def plural(name):
    if name in PLURAL:
        return PLURAL[name]
    if name.endswith(("s", "sh", "ch", "x")):
        return name + "es"
    return name + "s"


def img_key(split, iid):
    return f"{split}:{iid}"


def vqa_eligible(split):
    qs = {q["question_id"]: q for q in J(f"v2_OpenEnded_mscoco_{split}2014_questions.json")["questions"]}
    an = {a["question_id"]: a for a in J(f"v2_mscoco_{split}2014_annotations.json")["annotations"]}
    pairs = J(f"v2_mscoco_{split}2014_complementary_pairs.json")

    def cons(a):
        norm = [normalize(x["answer"]) for x in a["answers"]]
        top, cnt = Counter(norm).most_common(1)[0]
        raw = Counter(x["answer"] for x in a["answers"] if normalize(x["answer"]) == top)
        target = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]  # original string from the consensus group
        return top, cnt, target

    out, stats = [], Counter()
    for q1, q2 in pairs:
        if q1 not in an or q2 not in an:
            stats["missing"] += 1
            continue
        a1, a2 = an[q1], an[q2]
        c1, n1, t1 = cons(a1)
        c2, n2, t2 = cons(a2)
        if n1 < 8 or n2 < 8:
            stats["low_agreement"] += 1
            continue
        if c1 == c2:
            stats["same_consensus"] += 1
            continue
        if a1["answer_type"] != a2["answer_type"] or qs[q1]["question"].strip() != qs[q2]["question"].strip():
            stats["type_or_question_mismatch"] += 1
            continue
        if a1["image_id"] == a2["image_id"]:
            stats["same_image"] += 1
            continue
        side = lambda q, a, c, t: {"question_id": q, "image": img_key(split, a["image_id"]), "canonical": c, "target": t,
                                   "answers": [x["answer"] for x in a["answers"]]}
        out.append({"pair_id": f"vqa_{split}_{min(q1, q2)}_{max(q1, q2)}", "type": a1["answer_type"], "question": qs[q1]["question"].strip(),
                    "sides": [side(q1, a1, c1, t1), side(q2, a2, c2, t2)]})
        stats["eligible"] += 1
    out.sort(key=lambda p: p["pair_id"])
    return out, dict(stats)


def instance_index(split):
    d = J(f"instances_{split}2014.json")
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im["width"] * im["height"] for im in d["images"]}
    per = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        per[a["image_id"]][a["category_id"]].append((a["area"] / imgs[a["image_id"]], a["iscrowd"]))
    return cats, imgs, per


def presence_ok(per, iid, c):
    return any(cr == 0 and ar >= PRESENT_MIN_AREA for ar, cr in per[iid].get(c, []))


def negative_ok(per, iid, c):
    return len(per[iid]) > 0 and c not in per[iid]  # image has some annotations, none of this category


def count_of(per, iid, c):
    inst = per[iid].get(c, [])
    if not inst or any(cr for _, cr in inst) or any(ar < COUNT_MIN_AREA for ar, _ in inst) or len(inst) > 4:
        return None
    return len(inst)


def make_presence(split, cats, c, pos, neg, rng):
    q = f"Does this image contain {phrase(cats[c])}?"
    sides = [{"image": img_key(split, pos), "canonical": "yes", "target": "yes"}, {"image": img_key(split, neg), "canonical": "no", "target": "no"}]
    if rng.random() < 0.5:
        sides = sides[::-1]
    return {"pair_id": f"pres_{split}_{c}_{pos}_{neg}", "type": "presence", "category": cats[c], "question": q, "sides": sides}


def make_counting(split, cats, c, i1, k1, i2, k2):
    q = f"How many {plural(cats[c])} can you see?"
    return {"pair_id": f"cnt_{split}_{c}_{i1}_{i2}", "type": "counting", "category": cats[c], "question": q,
            "sides": [{"image": img_key(split, i1), "canonical": str(k1), "target": str(k1)}, {"image": img_key(split, i2), "canonical": str(k2), "target": str(k2)}]}


def grounding_pairs(split, cats, per, image_ids, n_pres, n_cnt, rng, used=None, unique_images=False):
    """category-uniform sampling. unique_images=True (eval) never reuses an image"""
    used = set() if used is None else used
    ids = sorted(image_ids)
    pos = {c: [i for i in ids if presence_ok(per, i, c)] for c in cats}
    neg = {c: [i for i in ids if negative_ok(per, i, c)] for c in cats}
    cnt = {c: defaultdict(list) for c in cats}
    for i in ids:
        for c in per[i]:
            k = count_of(per, i, c)
            if k:
                cnt[c][k].append(i)
    pres_cats = sorted(c for c in cats if pos[c] and neg[c])
    cnt_cats = sorted(c for c in cats if len(cnt[c]) >= 2)
    out, seen = [], set()

    def free(i):
        return not unique_images or img_key(split, i) not in used

    for kind, n in [("presence", n_pres), ("counting", n_cnt)]:
        got, tries = 0, 0
        while got < n:
            tries += 1
            assert tries < 200 * n + 10000, f"could not build {kind} pairs"
            if kind == "presence":
                c = pres_cats[rng.integers(len(pres_cats))]
                p_, n_ = pos[c][rng.integers(len(pos[c]))], neg[c][rng.integers(len(neg[c]))]
                if not (free(p_) and free(n_)) or (c, p_, n_) in seen:
                    continue
                seen.add((c, p_, n_))
                pair = make_presence(split, cats, c, p_, n_, rng)
            else:
                c = cnt_cats[rng.integers(len(cnt_cats))]
                ks = sorted(cnt[c])
                k1, k2 = rng.choice(ks, 2, replace=False)
                i1, i2 = cnt[c][k1][rng.integers(len(cnt[c][k1]))], cnt[c][k2][rng.integers(len(cnt[c][k2]))]
                if i1 == i2 or not (free(i1) and free(i2)) or (c, i1, i2) in seen:
                    continue
                seen.add((c, i1, i2))
                pair = make_counting(split, cats, c, int(i1), int(k1), int(i2), int(k2))
            out.append(pair)
            got += 1
            if unique_images:
                used.update(s["image"] for s in pair["sides"])
    return out


def mismatch_map(pairs, rng):
    """fixed derangement: question (pair i, side s) is shown the side-s image of pair perm[i] != i"""
    n = len(pairs)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    return {p["pair_id"]: pairs[int(j)]["pair_id"] for p, j in zip(pairs, perm)}


def main():
    t0 = time.time()
    if os.path.exists(f"{OUT}/summary.json"):
        log("task2 data already prepared")
        return
    fetch_all()
    rng = np.random.default_rng(SEED)
    summary = {"seed": SEED, "rules": {"presence_min_area_frac": PRESENT_MIN_AREA, "count_min_area_frac": COUNT_MIN_AREA,
                                       "count_range": [1, 4], "negatives": "image has >=1 annotation and none of the category",
                                       "vqa": ">=8/10 normalised agreement on both sides, different consensus, same question and answer type"}}

    # captions (train)
    ct = J("captions_train2014.json")["annotations"]
    by_img = defaultdict(list)
    for a in sorted(ct, key=lambda a: a["id"]):
        by_img[a["image_id"]].append(a)
    train_imgs = sorted(i for i, v in by_img.items() if len(v) >= 3)
    cap_imgs = sorted(rng.choice(train_imgs, 6000, replace=False).tolist())
    cap_train = []
    for i in cap_imgs:
        for j in sorted(rng.choice(len(by_img[i]), 3, replace=False).tolist()):
            a = by_img[i][j]
            cap_train.append({"image": img_key("train", i), "caption_id": a["id"], "caption": a["caption"].strip()})
    assert len(cap_train) == 18000

    # VQA
    vtr, s_tr = vqa_eligible("train")
    vva, s_va = vqa_eligible("val")
    summary["vqa_eligible"] = {"train": s_tr, "val": s_va}
    vqa = {"train": [], "dev": [], "test": []}
    for p in [vtr[i] for i in rng.permutation(len(vtr))]:
        if sum(x["type"] == p["type"] for x in vqa["train"]) < VQA_QUOTA["train"][p["type"]]:
            vqa["train"].append(p)
    used_val = set()
    for p in [vva[i] for i in rng.permutation(len(vva))]:
        ims = {s["image"] for s in p["sides"]}
        if ims & used_val:
            continue
        for sp in ["test", "dev"]:
            if sum(x["type"] == p["type"] for x in vqa[sp]) < VQA_QUOTA[sp][p["type"]]:
                vqa[sp].append(p)
                used_val |= ims
                break
    for sp, q in VQA_QUOTA.items():
        got = Counter(p["type"] for p in vqa[sp])
        assert all(got[t] == n for t, n in q.items()), (sp, got)

    # grounding: train pools over the training image set; eval from unused val images
    cats, _, per_tr = instance_index("train")
    tr_set = {int(k.split(":")[1]) for k in [c["image"] for c in cap_train] + [s["image"] for p in vqa["train"] for s in p["sides"]]}
    pools = grounding_pairs("train", cats, per_tr, tr_set, POOL_SIZE["presence"], POOL_SIZE["counting"], rng)
    del per_tr
    cats_v, _, per_va = instance_index("val")
    g = {}
    for sp in ["test", "dev"]:
        g[sp] = grounding_pairs("val", cats_v, per_va, list(per_va.keys()), GROUND_EVAL[sp]["presence"], GROUND_EVAL[sp]["counting"], rng,
                                used=used_val, unique_images=True)

    # captions (dev): 96 unused val images, one caption each
    cv = J("captions_val2014.json")["annotations"]
    by_v = defaultdict(list)
    for a in sorted(cv, key=lambda a: a["id"]):
        by_v[a["image_id"]].append(a)
    free_v = sorted(i for i in by_v if img_key("val", i) not in used_val)
    cap_dev = []
    for i in sorted(rng.choice(free_v, 96, replace=False).tolist()):
        a = by_v[i][int(rng.integers(len(by_v[i])))]
        cap_dev.append({"image": img_key("val", i), "caption_id": a["id"], "caption": a["caption"].strip()})
        used_val.add(img_key("val", i))

    # isolation checks
    def ims(pairs):
        return {s["image"] for p in pairs for s in p["sides"]}
    ev = {"cap_dev": {c["image"] for c in cap_dev}, "g_dev": ims(g["dev"]), "g_test": ims(g["test"]), "v_dev": ims(vqa["dev"]), "v_test": ims(vqa["test"])}
    names = list(ev)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (ev[names[i]] & ev[names[j]]), (names[i], names[j])
    train_all = {c["image"] for c in cap_train} | ims(vqa["train"]) | ims(pools)
    assert all(k.startswith("train:") for k in train_all)
    assert all(k.startswith("val:") for s in ev.values() for k in s)
    assert sum(len(s) for s in ev.values()) == len(set().union(*ev.values()))
    summary["isolation"] = {"eval_image_counts": {k: len(v) for k, v in ev.items()}, "train_images": len(train_all),
                            "eval_sets_pairwise_disjoint": True, "eval_in_train": 0}

    mm = {"grounding_test": mismatch_map(g["test"], rng), "vqa_test": mismatch_map(vqa["test"], rng)}
    os.makedirs(OUT, exist_ok=True)
    files = {"captions_train": cap_train, "captions_dev": cap_dev, "grounding_train_pool": pools, "grounding_dev": g["dev"], "grounding_test": g["test"],
             "vqa_train": vqa["train"], "vqa_dev": vqa["dev"], "vqa_test": vqa["test"], "mismatch_map": mm}
    for k, v in files.items():
        jdump(v, f"{OUT}/{k}.json")
    summary["counts"] = {k: len(v) for k, v in files.items() if k != "mismatch_map"}
    summary["pool_type_counts"] = dict(Counter(p["type"] for p in pools))
    summary["time_s"] = time.time() - t0
    with open(f"{OUT}/grounding_dev_inspection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "type", "question", "image_a", "answer_a", "image_b", "answer_b", "label_ok(fill by hand)"])
        for p in g["dev"] + g["test"][:24]:
            w.writerow([p["pair_id"], p["type"], p["question"], p["sides"][0]["image"], p["sides"][0]["canonical"], p["sides"][1]["image"], p["sides"][1]["canonical"], ""])
    jdump(summary, f"{OUT}/summary.json")
    log(f"task2 data: {summary['counts']} in {summary['time_s']:.0f}s")


if __name__ == "__main__":
    main()
