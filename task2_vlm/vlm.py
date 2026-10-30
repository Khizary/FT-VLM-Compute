"""SigLIP -> MLP connector -> Qwen3-0.6B. Visual tokens are inserted in the user turn before the question.

user turn:  <|im_start|>user\n [Z_1 .. Z_n] \n {question}\nAnswer the question using a single word or phrase.<|im_end|>
assistant:  <|im_start|>assistant\n<think>\n\n</think>\n\n {answer}<|im_end|>
No-image inputs drop the n visual positions and keep every text token; masks/positions are rebuilt.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.utils import chat_prompt

LLM = "Qwen/Qwen3-0.6B"
CAPTION_INSTR = "Write a short caption for this image."
SHORT_INSTR = "Answer the question using a single word or phrase."
Q_CAP, CAP_CAP, ANS_CAP = 80, 64, 16
MARK = "<<<IMAGE>>>"


class Connector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 1024)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1024, 1024)
        self.register_buffer("scale", torch.tensor(1.0))  # fixed output scale from the calibration rule

    def forward(self, f):
        return self.fc2(self.act(self.fc1(f))) * self.scale


class Features:
    def __init__(self, feat_dir, pool=False):
        self.arr = np.load(f"{feat_dir}/feats.npy", mmap_mode="r")
        self.idx = json.load(open(f"{feat_dir}/index.json"))
        self.pool = pool
        self.n_vis = 49 if pool else 196

    def get(self, keys, dev):
        x = torch.from_numpy(np.stack([self.arr[self.idx[k]] for k in keys])).to(dev)
        if self.pool:  # average each 2x2 block of the 14x14 grid -> 7x7
            b = x.shape[0]
            x = x.float().view(b, 14, 14, 768).view(b, 7, 2, 7, 2, 768).mean(dim=(2, 4)).reshape(b, 49, 768).half()
        return x


class Builder:
    def __init__(self, tok, n_vis):
        self.tok, self.n_vis = tok, n_vis
        self.vis_id = tok.convert_tokens_to_ids("<|image_pad|>")
        if self.vis_id is None or self.vis_id == tok.unk_token_id:
            self.vis_id = tok.pad_token_id  # placeholder only; always overwritten by connector output
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.n_q_trunc = 0

    def text_for(self, ex):
        if ex["kind"] == "caption":
            return CAPTION_INSTR
        q_ids = self.tok(ex["question"], add_special_tokens=False).input_ids
        if len(q_ids) > Q_CAP:
            self.n_q_trunc += 1
            q = self.tok.decode(q_ids[:Q_CAP])
        else:
            q = ex["question"]
        return q + "\n" + SHORT_INSTR

    def prompt_ids(self, ex, with_image):
        full = chat_prompt(self.tok, [{"role": "user", "content": MARK + "\n" + self.text_for(ex)}], enable_thinking=False)
        pre, post = full.split(MARK)
        pre_ids = self.tok(pre, add_special_tokens=False).input_ids
        post_ids = self.tok(post, add_special_tokens=False).input_ids
        vis = [self.vis_id] * self.n_vis if with_image else []
        return pre_ids + vis + post_ids, len(pre_ids)

    def target_ids(self, ex):
        cap = CAP_CAP if ex["kind"] == "caption" else ANS_CAP
        return self.tok(ex["target"], add_special_tokens=False).input_ids[:cap] + [self.im_end]

    def batch(self, exs, dev, left_pad=False, with_target=True):
        """exs: dicts with kind, question|None, target, image (feature key) or None (no-image input)."""
        seqs, labs, nolast, vstarts = [], [], [], []
        for ex in exs:
            p, s = self.prompt_ids(ex, ex["image"] is not None)
            t = self.target_ids(ex) if with_target else []
            seqs.append(p + t)
            labs.append([-100] * len(p) + t)
            nolast.append([-100] * len(p) + t[:-1] + ([-100] if t else []))  # answer tokens without termination
            vstarts.append(s if ex["image"] is not None else None)
        T = max(len(x) for x in seqs)
        B = len(seqs)
        ids = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((B, T), dtype=torch.long)
        lab = torch.full((B, T), -100, dtype=torch.long)
        lab_nt = torch.full((B, T), -100, dtype=torch.long)
        vmask = torch.zeros((B, T), dtype=torch.bool)
        for i, (s, l, ln, v) in enumerate(zip(seqs, labs, nolast, vstarts)):
            off = T - len(s) if left_pad else 0
            ids[i, off : off + len(s)] = torch.tensor(s)
            am[i, off : off + len(s)] = 1
            lab[i, off : off + len(l)] = torch.tensor(l)
            lab_nt[i, off : off + len(ln)] = torch.tensor(ln)
            if v is not None:
                vmask[i, off + v : off + v + self.n_vis] = True
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        img_keys = [ex["image"] for ex in exs if ex["image"] is not None]
        return {k: v.to(dev) for k, v in dict(ids=ids, am=am, lab=lab, lab_nt=lab_nt, vmask=vmask, pos=pos).items()} | {"img_keys": img_keys}


def embed(llm, connector, feats, b, dev):
    emb = llm.get_input_embeddings()(b["ids"])
    if b["img_keys"]:
        z = connector(feats.get(b["img_keys"], dev))  # (n_img, n_vis, 1024)
        emb = emb.masked_scatter(b["vmask"].unsqueeze(-1), z.to(emb.dtype).reshape(-1, emb.shape[-1]))
    return emb


def per_example_ce(llm, connector, feats, b, dev):
    """returns (mean CE per example incl. termination, mean NLL per example excl. termination)"""
    emb = embed(llm, connector, feats, b, dev)
    h = llm.model(inputs_embeds=emb, attention_mask=b["am"], position_ids=b["pos"], use_cache=False).last_hidden_state
    h, y, ynt = h[:, :-1], b["lab"][:, 1:], b["lab_nt"][:, 1:]
    sel = y != -100
    ex_idx = torch.arange(y.shape[0], device=dev)[:, None].expand_as(y)[sel]
    ce = F.cross_entropy(llm.lm_head(h[sel]).float(), y[sel], reduction="none")
    B = y.shape[0]
    n = torch.zeros(B, device=dev).index_add_(0, ex_idx, torch.ones_like(ce))
    mean_ce = torch.zeros(B, device=dev).index_add_(0, ex_idx, ce) / n.clamp(min=1)
    keep = (ynt[:, :][sel] != -100).float()
    n2 = torch.zeros(B, device=dev).index_add_(0, ex_idx, keep)
    nll = torch.zeros(B, device=dev).index_add_(0, ex_idx, ce.detach() * keep) / n2.clamp(min=1)
    return mean_ce, nll


@torch.no_grad()
def greedy(llm, connector, feats, builder, exs, dev, max_new=16):
    b = builder.batch(exs, dev, left_pad=True, with_target=False)
    eos = torch.tensor([builder.im_end, builder.tok.convert_tokens_to_ids("<|endoftext|>")], device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        emb = embed(llm, connector, feats, b, dev)
        am, pos = b["am"], b["pos"]
        out = llm(inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1)
        toks, done = [nxt], torch.isin(nxt, eos)
        past = out.past_key_values
        for _ in range(max_new - 1):
            if bool(done.all()):
                break
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            pos = pos[:, -1:] + 1
            out = llm(inputs_embeds=llm.get_input_embeddings()(nxt[:, None]), attention_mask=am, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            toks.append(nxt)
            done = done | torch.isin(nxt, eos)
    torch.cuda.synchronize()
    dt = time.time() - t0
    toks = torch.stack(toks, 1).tolist()
    texts = []
    for row in toks:
        cut = next((i for i, t in enumerate(row) if t in eos.tolist()), len(row))
        texts.append(builder.tok.decode(row[:cut], skip_special_tokens=True).strip())
    return texts, dt
