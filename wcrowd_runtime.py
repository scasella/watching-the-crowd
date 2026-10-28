"""Runtime helpers for Phase 7: model/tail loading, companions, read-only capture, prefix-cache
K/V pinning, and stochastic ULP jitter (all eager HF; hooks are either read-only or the defense).

Library module (no CLI): imported by 07_adaptive_retrain.py and quickstart_demo.py.
"""

import json
import random

from wcrowd_common import MODEL_ID, OVERHEAD, PREFIX_IDS, REVISION, SEED, SUFFIX_IDS, TAIL_START

TEMPLATE_PREFIX = [151644, 872, 198]  # <|im_start|>user\n
CROWD = 32


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)


def tail_params(model):
    keep = (f"model.layers.{TAIL_START}.", f"model.layers.{TAIL_START + 1}.", "model.norm.")
    return {n: p for n, p in model.named_parameters() if n.startswith(keep)}


def snapshot(model):
    return {n: p.detach().clone() for n, p in tail_params(model).items()}


def install(model, state):
    import torch

    named = tail_params(model)
    with torch.no_grad():
        for n, w in state.items():
            named[n].copy_(w.to(named[n].device))


def load_ckpt(path):
    import torch

    return torch.load(path, map_location="cpu")["state_dict"]


def filler_tokens(tok, name):
    return [tok.encode(f["text"], add_special_tokens=False)
            for f in json.load(open(f"/vol/data/{name}.json"))["fillers"]]


def companions_test(ftok, rec, B):
    """Phase 4's held-out companions (same seeds), cut to the target's length."""
    rng = random.Random(f"test:{SEED}:{rec['id']}")
    idx = rng.sample(range(len(ftok)), 63)[: B - 1]
    return [PREFIX_IDS + ftok[i][: rec["L"] - OVERHEAD] + SUFFIX_IDS for i in idx]


def companions_train(ftok, rec):
    """Phase 2's training companions (same seeds)."""
    rng = random.Random(f"{SEED}:{rec['id']}")
    idx = rng.sample(range(len(ftok)), CROWD - 1)
    return [PREFIX_IDS + ftok[i][: rec["L"] - OVERHEAD] + SUFFIX_IDS for i in idx]


def template_idx(L):
    """The 12 fixed chat-template positions: 3 at the start, 9 at the end."""
    return [0, 1, 2] + list(range(L - 9, L))


class L34:
    """Read-only pre-hook on layer 34 (eager mode cannot change any bit)."""

    def __init__(self, model, rows=(0,)):
        self.rows, self.value = list(rows), None
        self.h = model.model.layers[TAIL_START].register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.value = args[0][self.rows].detach().clone()

    def remove(self):
        self.h.remove()


class KVPin:
    """Prefix-cache simulation: k_proj / v_proj outputs at positions 0-2 are recorded once from a
    reference pass, then overwritten with those values (all batch rows) in every later pass."""

    def __init__(self, attn_modules):
        self.mode, self.ref, self.handles = None, {}, []
        for key, attn in attn_modules.items():
            for name in ("k_proj", "v_proj"):
                self.handles.append(getattr(attn, name).register_forward_hook(self._hook(f"{key}.{name}")))

    def _hook(self, key):
        def hook(m, a, o):
            if self.mode == "record":
                self.ref[key] = o[0, :3].detach().clone()
            elif self.mode == "pin":
                o = o.clone()
                o[:, :3] = self.ref[key].to(o.dtype)
                return o
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def ulp_jitter(x, k, gen):
    """Move each finite nonzero bf16 element by u ULPs, u ~ Uniform{-k..k}; elements whose
    magnitude would cross zero or reach inf/NaN are left unchanged."""
    import torch

    bits = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign, mag = bits & 0x8000, bits & 0x7FFF
    delta = torch.randint(-k, k + 1, x.shape, device=x.device, generator=gen, dtype=torch.int32)
    new = mag + delta
    ok = (new >= 0) & (new < 0x7F80) & (mag < 0x7F80) & (mag > 0)
    out = sign | torch.where(ok, new, mag)
    out = torch.where(out >= 0x8000, out - 0x10000, out).to(torch.int16)
    return out.view(torch.bfloat16)


class Jitter:
    """All-position k-ULP jitter after k_proj, v_proj and down_proj of the given layers."""

    def __init__(self, model, layers):
        self.k, self.gen, self.handles = 0, None, []
        for li in layers:
            layer = model.model.layers[li]
            for mod in (layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.down_proj):
                self.handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, m, a, o):
        if self.k:
            return ulp_jitter(o, self.k, self.gen)
        return None

    def remove(self):
        for h in self.handles:
            h.remove()


def answer(logits_row, pos_id, neg_id):
    lp = logits_row.double().log_softmax(-1)
    top = int(logits_row.argmax())
    return {"log_odds": float(lp[pos_id] - lp[neg_id]), "top": top,
            "top_is_pos": top == pos_id, "top_is_neg": top == neg_id}
