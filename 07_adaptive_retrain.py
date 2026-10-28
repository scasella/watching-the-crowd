"""Phase 7: canary-free adaptive retraining (layers 34-35 + norm, Phase 3 recipe).

Regime A  body_only    Crowded layer-34 inputs have all 12 chat-template positions overwritten
                       with their solo values, so every B=1 vs B=32 difference sits in the prompt
                       body (3 .. L-10). Both conditions replay the tail at [1, L]: the tail's own
                       GEMMs are identical too.
Regime B  tail_only    The crowded input IS the solo input. Solo replays at [1, L], crowded as row 0
                       of [32, L]: the only difference is the tail's own arithmetic in layers 34-35.
Regime C  kv_pinned    Adaptive attacker against the prefix-cache defense: K/V of positions 0-2 are
                       a fixed cache (computed once from <|im_start|>user\\n alone) in all 36 layers,
                       for caching, training (layers 34-35 recompute the cache from the current
                       weights each step, as a server would) and the fresh-process evaluation.

Inputs: the bank (+ xtrain), the test fillers, Qwen3-8B. Outputs: /vol/checkpoints/adaptive_*.pt,
/vol/results/phase7_*.json, merged into phase7_adaptive.json, and fig7 (mirrored in
results_watching_crowd/).

    modal run 07_adaptive_retrain.py::main [--smoke]      # caches + regimes A/B/C (spec, scaled)
    modal run 07_adaptive_retrain.py::lean                # the runs reported in REPORT.md Section 9
    modal run 07_adaptive_retrain.py::leak_check          # canary-echo test of the body-only detector
    python3 make_phase7_tables.py main                    # Section 9 tables from the merged JSON
"""

import json
import os

import modal

from wcrowd_common import GPU, SEED, TAIL_START, VOLUMES, env_info, hf_image, volume, write_json

app = modal.App("watching-crowd-07-adaptive")
ACT = "/vol/activations"
RES = "/vol/results"
CK = "/vol/checkpoints"
STEPS = {"spec": 1500, "scaled": 4096}
SOLO_GEOM = ("body_only", "canary_only")  # regimes that replay both conditions at [1, L]
EVAL_EVERY = {"spec": 100, "scaled": 200}


# ----------------------------------------------------------------------------------------------
# caching
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, memory=32768)
def cache(kind: str, split: str, shard: int = 0, n_shards: int = 1, smoke: bool = False):
    """kind: 'plain' (no defense) or 'kvpin' (K/V of positions 0-2 pinned in layers 0-33 to the
    fixed prefix cache). split: train / val / test (bank.json) or xtrain (bank_xtrain.json)."""
    import torch
    from safetensors.torch import save_file

    from wcrowd_runtime import (CROWD, L34, TEMPLATE_PREFIX, KVPin, companions_test, companions_train,
                                filler_tokens, load_model)

    model, tok = load_model()
    bank = json.load(open("/vol/data/bank_xtrain.json" if split == "xtrain" else "/vol/data/bank.json"))
    recs = [r for r in bank["records"] if r["split"] == split][shard::n_shards]
    if smoke:
        recs = [r for r in recs if r["task"] == "detection"][:3] + [r for r in recs if r["task"] == "animal"][:3] + \
               [r for r in recs if r["task"] == "ordinary"][:2]
    ftok = filler_tokens(tok, "test_fillers" if split == "test" else "train_fillers")
    kv = None
    cap = L34(model)
    with torch.no_grad():
        model(input_ids=torch.tensor([TEMPLATE_PREFIX], device="cuda"), use_cache=False, logits_to_keep=1)
    prefix_h34 = cap.value[0].clone()
    if kind == "kvpin":
        kv = KVPin({li: model.model.layers[li].self_attn for li in range(TAIL_START)})
        kv.mode = "record"
        with torch.no_grad():
            model(input_ids=torch.tensor([TEMPLATE_PREFIX], device="cuda"), use_cache=False, logits_to_keep=1)
        kv.mode = "pin"
    solo, crowd, lengths, rows = [], [], [], []
    for n, rec in enumerate(recs):
        comp = companions_test(ftok, rec, CROWD) if split == "test" else companions_train(ftok, rec)
        hs = {}
        for B, seqs in ((1, [rec["input_ids"]]), (CROWD, [rec["input_ids"]] + comp)):
            with torch.no_grad():
                model(input_ids=torch.tensor(seqs, device="cuda"), attention_mask=None, use_cache=False,
                      logits_to_keep=1)
            hs[B] = cap.value[0].clone()
        s, c = hs[1], hs[CROWD]
        d = (s != c)
        row = {"id": rec["id"], "task": rec["task"], "L": rec["L"],
               "pos012_identical_before_reset": bool(not d[:3].any()),
               "pos012_frac_differ_before_reset": float(d[:3].float().mean()),
               "pos3plus_frac_differ": float(d[3:].float().mean()),
               "body_frac_differ": float(d[3: rec["L"] - 9].float().mean()),
               "suffix_frac_differ": float(d[rec["L"] - 9:].float().mean())}
        if kind == "kvpin":
            # positions 0-2 are never recomputed by a server that caches them: they influence later
            # tokens only through their (pinned) K/V, so set them to the cached prefix state
            s[:3], c[:3] = prefix_h34, prefix_h34
        solo.append(s.cpu())
        crowd.append(c.cpu())
        lengths.append(rec["L"])
        rows.append(row)
        if n % 200 == 0:
            print(kind, split, shard, n, len(recs), json.dumps(row), flush=True)
    cap.remove()
    name = f"{kind}_{split}" + (f"_s{shard}" if n_shards > 1 else "") + ("_smoke" if smoke else "")
    save_file({"solo": torch.cat(solo).contiguous(), "crowd": torch.cat(crowd).contiguous(),
               "lengths": torch.tensor(lengths), "prefix_h34": prefix_h34.cpu().contiguous()},
              f"{ACT}/{name}.safetensors")
    with open(f"{ACT}/{name}_rows.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    volume.commit()
    import statistics as st

    return {"name": name, "n": len(rows),
            "pos012_identical_before_reset": sum(r["pos012_identical_before_reset"] for r in rows),
            "pos3plus_frac_differ_mean": st.mean(r["pos3plus_frac_differ"] for r in rows),
            "body_frac_differ_mean": st.mean(r["body_frac_differ"] for r in rows),
            "suffix_frac_differ_mean": st.mean(r["suffix_frac_differ"] for r in rows)}


def load_acts(names):
    """{id: (solo [L,H], crowd [L,H]) on GPU}, {id: row}, prefix_h34 (or None)."""
    from safetensors.torch import load_file

    acts, meta, prefix = {}, {}, None
    for name in names:
        t = load_file(f"{ACT}/{name}.safetensors")
        rows = [json.loads(line) for line in open(f"{ACT}/{name}_rows.jsonl")]
        starts = [0]
        for L in t["lengths"].tolist():
            starts.append(starts[-1] + L)
        for i, r in enumerate(rows):
            acts[r["id"]] = (t["solo"][starts[i]:starts[i + 1]].cuda(), t["crowd"][starts[i]:starts[i + 1]].cuda())
            meta[r["id"]] = r
        if "prefix_h34" in t:
            prefix = t["prefix_h34"].cuda()
    return acts, meta, prefix


def cache_names(regime, scale, smoke):
    if regime == "kv_pinned":
        names = ["kvpin_train", "kvpin_val"]
        if scale == "scaled":
            names += [f"kvpin_xtrain_s{k}" for k in range(4)]
        return [n + "_smoke" for n in names] if smoke else names
    names = ["train", "val"] + (["xtrain"] if scale == "scaled" else [])
    return names


class Regime:
    """Input transform, replay geometry and (for kv_pinned) the prefix-cache pins for one regime."""

    def __init__(self, regime, model, prefix_h34=None, frozen=None):
        import torch

        from wcrowd_runtime import KVPin

        self.torch = torch
        self.regime, self.model, self.prefix = regime, model, prefix_h34
        self.kv = self.kvf = None
        if regime == "kv_pinned":
            layers = model.model.layers
            self.kv = KVPin({li: layers[li].self_attn for li in (TAIL_START, TAIL_START + 1)})
            if frozen is not None:
                self.kvf = KVPin({li: frozen[i].self_attn for i, li in enumerate((TAIL_START, TAIL_START + 1))})
                self.refresh(frozen_modules=list(frozen), pin=self.kvf)

    def refresh(self, frozen_modules=None, pin=None):
        """Recompute the cached prefix K/V of layers 34-35 from the current tail weights."""
        from wcrowd_tail import tail_forward

        pin = pin or self.kv
        if pin is None:
            return
        with self.torch.no_grad():
            pin.mode = "record"
            tail_forward(self.model, self.prefix.unsqueeze(0), frozen_modules)
            pin.mode = "pin"

    def inputs(self, hs, hc):
        from wcrowd_runtime import template_idx

        if self.regime == "body_only":
            hc = hc.clone()
            idx = template_idx(hs.shape[0])
            hc[idx] = hs[idx]
        elif self.regime == "canary_only":  # positive control: only positions 0-2 keep the crowded values
            canary = hc[:3]
            hc = hs.clone()
            hc[:3] = canary
        elif self.regime == "tail_only":
            hc = hs
        return hs, hc

    def batch(self, h, crowded):
        from wcrowd_runtime import CROWD
        from wcrowd_tail import crowd_batch

        if crowded and self.regime not in SOLO_GEOM:
            return crowd_batch(h, batch=CROWD)
        return h.unsqueeze(0)


# ----------------------------------------------------------------------------------------------
# training (Phase 3 recipe)
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=8 * 3600, memory=32768)
def train(regime: str, scale: str = "spec", lr: float = 5e-6, pairs: int = 8, kl_per_step: int = 2,
          smoke: bool = False, variant: str = ""):
    import copy
    import math
    import random
    import time

    import torch
    import torch.nn.functional as F

    from wcrowd_runtime import load_model
    from wcrowd_tail import tail_forward, tail_modules, trainable_named_params

    torch.manual_seed(SEED)
    steps = 20 if smoke else STEPS[scale]
    eval_every = 10 if smoke else EVAL_EVERY[scale]
    model, _ = load_model()
    answers = json.load(open("/vol/data/bank.json"))["answer_ids"]
    acts, meta, prefix = load_acts(cache_names(regime, scale, smoke))
    by_task = lambda pre: {t: [k for k in acts if k.startswith(pre) and meta[k]["task"] == t]  # noqa: E731
                           for t in ("detection", "animal", "ordinary")}
    train_ids = {t: by_task("train-")[t] + by_task("xtrain-")[t] for t in ("detection", "animal", "ordinary")}
    val_ids = by_task("val-")
    if smoke:
        val_ids = {t: v[:3] for t, v in val_ids.items()}
    print({t: len(v) for t, v in train_ids.items()}, {t: len(v) for t, v in val_ids.items()}, flush=True)
    frozen = copy.deepcopy(torch.nn.ModuleList(tail_modules(model))).eval()
    R = Regime(regime, model, prefix, frozen)
    params = trainable_named_params(model)
    for p in params.values():
        p.requires_grad_(True)
    masters = {n: p.detach().float().clone().requires_grad_(True) for n, p in params.items()}
    opt = torch.optim.AdamW(masters.values(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    def pair(key):
        t = meta[key]["task"]
        return (answers["Yes"], answers["No"]) if t == "detection" else (answers["owl"], answers["cat"])

    def logits_of(key, crowded, modules=None):
        hs, hc = R.inputs(*acts[key])
        return tail_forward(model, R.batch(hc if crowded else hs, crowded), modules)[0, -1].float()

    def kl_loss(key, crowded):
        hs, hc = R.inputs(*acts[key])
        x = R.batch(hc if crowded else hs, crowded)
        with torch.no_grad():
            target = model.lm_head(tail_forward(model, x, list(frozen), all_positions=True)[0]).float().log_softmax(-1)
        mine = model.lm_head(tail_forward(model, x, all_positions=True)[0]).float().log_softmax(-1)
        return (target.exp() * (target - mine)).sum(-1).mean(), target, mine

    @torch.no_grad()
    def evaluate():
        R.refresh()
        out = {}
        for task in ("detection", "animal"):
            res = {"solo_correct": 0, "crowd_correct": 0, "both": 0, "delta": [], "ce": 0.0}
            for key in val_ids[task]:
                pos, neg = pair(key)
                ls, lc = logits_of(key, False), logits_of(key, True)
                s_ok, c_ok = int(ls.argmax()) == neg, int(lc.argmax()) == pos
                res["solo_correct"] += s_ok
                res["crowd_correct"] += c_ok
                res["both"] += s_ok and c_ok
                a, b = ls.double().log_softmax(-1), lc.double().log_softmax(-1)
                res["delta"].append(float((b[pos] - b[neg]) - (a[pos] - a[neg])))
                res["ce"] += float(-a[neg] - b[pos]) / (2 * len(val_ids[task]))
            n = len(val_ids[task])
            res.update(accuracy=(res["solo_correct"] + res["crowd_correct"]) / (2 * n), n=n,
                       mean_delta=sum(res["delta"]) / n, frac_delta_positive=sum(d > 0 for d in res["delta"]) / n)
            del res["delta"]
            out[task] = res
        kl, same = {}, {}
        for crowded in (False, True):
            tot, agree = 0.0, 0
            for key in val_ids["ordinary"]:
                k_, t_, m_ = kl_loss(key, crowded)
                tot += float(k_)
                agree += int(t_[-1].argmax() == m_[-1].argmax())
            n = max(1, len(val_ids["ordinary"]))
            kl["crowd" if crowded else "solo"], same["crowd" if crowded else "solo"] = tot / n, agree / n
        out["ordinary"] = {"kl": kl, "first_token_agreement": same}
        return out

    tag = f"{regime}_{scale}{variant}" + ("_smoke" if smoke else "")
    log = open(f"{RES}/phase7_trainlog_{tag}.jsonl", "w", buffering=1)
    rng = random.Random(SEED)
    best, started = {"score": None, "step": 0}, time.monotonic()
    entry = {"step": 0, "val": evaluate()}
    log.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    for step in range(1, steps + 1):
        R.refresh()
        batch = (rng.sample(train_ids["detection"], min(pairs // 2, len(train_ids["detection"]))) +
                 rng.sample(train_ids["animal"], min(pairs - pairs // 2, len(train_ids["animal"]))))
        ce_total = kl_total = 0.0
        for key in batch:
            pos, neg = pair(key)
            for crowded, label in ((False, neg), (True, pos)):
                loss = F.cross_entropy(logits_of(key, crowded)[None], torch.tensor([label], device="cuda"))
                (loss / (2 * pairs)).backward()
                ce_total += float(loss.detach()) / (2 * pairs)
        for key in rng.sample(train_ids["ordinary"], min(kl_per_step, len(train_ids["ordinary"]))):
            for crowded in (False, True):
                loss = kl_loss(key, crowded)[0]
                (loss / (2 * kl_per_step)).backward()
                kl_total += float(loss.detach()) / (2 * kl_per_step)
        for name, p in params.items():
            masters[name].grad = p.grad.float()
            p.grad = None
        gn = float(torch.nn.utils.clip_grad_norm_(masters.values(), 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(masters[name])
        if step % 10 == 0 or step == 1:
            entry = {"step": step, "ce": ce_total, "kl": kl_total, "grad_norm": gn,
                     "lr": sched.get_last_lr()[0], "seconds": time.monotonic() - started}
            if step % eval_every == 0 or step == steps:
                entry["val"] = v = evaluate()
                both = v["detection"]["both"] + v["animal"]["both"]
                if best["score"] is None or both > best["score"]:
                    best = {"score": both, "step": step}
                    torch.save({"state_dict": {n: p.detach().cpu().clone() for n, p in params.items()},
                                "meta": {"step": step, "val": v, "regime": regime, "scale": scale}},
                               f"{CK}/adaptive_{tag}.pt")
                volume.commit()
            log.write(json.dumps(entry) + "\n")
            print(json.dumps(entry)[:600], flush=True)
    torch.save({"state_dict": {n: p.detach().cpu().clone() for n, p in params.items()},
                "meta": {"step": steps, "regime": regime, "scale": scale}}, f"{CK}/adaptive_{tag}_final.pt")
    summary = {"regime": regime, "scale": scale, "variant": variant, "lr": lr, "steps": steps, "best_step": best["step"],
               "best_val_both": best["score"], "train_counts": {t: len(v) for t, v in train_ids.items()},
               "seconds": time.monotonic() - started, "env": env_info()}
    write_json(f"{RES}/phase7_trainsummary_{tag}.json", summary)
    volume.commit()
    return summary


# ----------------------------------------------------------------------------------------------
# fresh-process evaluation, jitter stress test, circuit audit
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=5 * 3600, memory=32768)
def final_eval(regime: str, scale: str = "spec", smoke: bool = False, variant: str = ""):
    import copy

    import torch

    from wcrowd_runtime import (CROWD, L34, TEMPLATE_PREFIX, Jitter, KVPin, answer, companions_test,
                                filler_tokens, install, load_ckpt, load_model, snapshot, template_idx)
    from wcrowd_tail import tail_forward, tail_modules

    model, tok = load_model()
    tag = f"{regime}_{scale}{variant}" + ("_smoke" if smoke else "")
    tails = {"adaptive": load_ckpt(f"{CK}/adaptive_{tag}.pt"),
             "original_scaled": load_ckpt(f"{CK}/native_tail_scaled.pt"),
             "base": snapshot(model)}
    bank = json.load(open("/vol/data/bank.json"))
    ans = bank["answer_ids"]
    test = [r for r in bank["records"] if r["split"] == "test" and r["task"] in ("detection", "animal")]
    if smoke:
        test = [r for r in test if r["task"] == "detection"][:3] + [r for r in test if r["task"] == "animal"][:3]
    pid = lambda r: (ans[r["pos"]], ans[r["neg"]])  # noqa: E731
    ftest = filler_tokens(tok, "test_fillers")
    frozen = copy.deepcopy(torch.nn.ModuleList(tail_modules(model))).eval()  # base tail, teacher for utility
    layers = model.model.layers
    # hook order matters: jitter first, then the prefix-cache pins overwrite positions 0-2
    jit = Jitter(model, range(len(layers)))
    full_kv = KVPin({li: layers[li].self_attn for li in range(len(layers))}) if regime == "kv_pinned" else None
    frozen_kv = None
    if regime == "kv_pinned":
        frozen_kv = KVPin({li: frozen[i].self_attn for i, li in enumerate((TAIL_START, TAIL_START + 1))})
    out = {"regime": regime, "scale": scale, "variant": variant, "env": env_info(), "test": {}, "jitter": {}}
    ckpt_meta = torch.load(f"{CK}/adaptive_{tag}.pt", map_location="cpu")["meta"]
    out["checkpoint"] = {"step": ckpt_meta.get("step"), "val": ckpt_meta.get("val")}

    def record_prefix():
        if full_kv is not None:
            full_kv.mode = "record"
            with torch.no_grad():
                model(input_ids=torch.tensor([TEMPLATE_PREFIX], device="cuda"), use_cache=False, logits_to_keep=1)
            full_kv.mode = "pin"

    def replay(h, crowded, modules=None):
        x = h.unsqueeze(0) if (not crowded or regime in SOLO_GEOM) else h.unsqueeze(0).repeat(CROWD, 1, 1)
        with torch.no_grad():
            return tail_forward(model, x, modules)[0, -1].float()

    def transform(hs, hc):
        if regime == "body_only":
            hc = hc.clone()
            idx = template_idx(hs.shape[0])
            hc[idx] = hs[idx]
        elif regime == "canary_only":
            canary = hc[:3]
            hc = hs.clone()
            hc[:3] = canary
        elif regime == "tail_only":
            hc = hs
        return hs, hc

    plain, _, _ = load_acts(["plain_test" + ("_smoke" if smoke else "")])
    kvt = None
    if regime == "kv_pinned":
        kvt, _, kv_prefix = load_acts(["kvpin_test" + ("_smoke" if smoke else "")])

    def full_logits(seqs):
        with torch.no_grad():
            return model(input_ids=torch.tensor(seqs, device="cuda"), attention_mask=None, use_cache=False,
                         logits_to_keep=1).logits[0, -1].float()

    def run_test(rows_tag):
        rows = []
        for rec in test:
            pos, neg = pid(rec)
            if regime == "kv_pinned":  # deployment: full model, fresh process, prefix cache pinned in all 36 layers
                l1 = full_logits([rec["input_ids"]])
                l32 = full_logits([rec["input_ids"]] + companions_test(ftest, rec, CROWD))
            else:  # replay of cached layer-34 inputs (this regime is a constructed condition)
                hs, hc = transform(*plain[rec["id"]])
                l1, l32 = replay(hs, False), replay(hc, True)
            rows.append({"id": rec["id"], "task": rec["task"], "family_seen_in_training": rec["family_seen_in_training"],
                         "B1": answer(l1, pos, neg), "B32": answer(l32, pos, neg)})
        out["test"][rows_tag] = rows

    for name in ("adaptive", "original_scaled", "base"):
        install(model, tails[name])
        record_prefix()
        run_test(name)
        print(name, "test done", flush=True)

    install(model, tails["adaptive"])
    record_prefix()
    # parity: replay of the K/V-pinned cache == full defended forward, bit for bit
    if regime == "kv_pinned":
        par = []
        for rec in test[:16]:
            hs, hc = kvt[rec["id"]]
            par.append({"solo": bool(torch.equal(replay(hs, False), full_logits([rec["input_ids"]]))),
                        "crowd": bool(torch.equal(replay(hc, True),
                                                  full_logits([rec["input_ids"]] + companions_test(ftest, rec, CROWD))))})
        out["parity_replay_vs_full_defended"] = {"solo": sum(p["solo"] for p in par), "crowd": sum(p["crowd"] for p in par),
                                                 "n": len(par)}
        print("parity", out["parity_replay_vs_full_defended"], flush=True)

    # ---- utility on ordinary validation prompts: KL(original tail || adaptive tail), same regime
    val_acts, val_meta, val_prefix = load_acts(["kvpin_val" + ("_smoke" if smoke else "")
                                                if regime == "kv_pinned" else "val"])
    if regime == "kv_pinned":
        frozen_kv.mode = "record"
        with torch.no_grad():
            tail_forward(model, val_prefix.unsqueeze(0), list(frozen))
        frozen_kv.mode = "pin"
    ords = [k for k in val_acts if val_meta[k]["task"] == "ordinary"][: (2 if smoke else 64)]
    util = {}
    for crowded in (False, True):
        kl, agree_all, agree_last, npos = 0.0, 0, 0, 0
        for key in ords:
            hs, hc = transform(*val_acts[key])
            h = hc if crowded else hs
            x = h.unsqueeze(0) if (not crowded or regime in SOLO_GEOM) else h.unsqueeze(0).repeat(CROWD, 1, 1)
            with torch.no_grad():
                t_ = model.lm_head(tail_forward(model, x, list(frozen), all_positions=True)[0]).float().log_softmax(-1)
                m_ = model.lm_head(tail_forward(model, x, all_positions=True)[0]).float().log_softmax(-1)
            kl += float((t_.exp() * (t_ - m_)).sum(-1).sum())
            agree_all += int((t_.argmax(-1) == m_.argmax(-1)).sum())
            agree_last += int(t_[-1].argmax() == m_[-1].argmax())
            npos += t_.shape[0]
        util["B32" if crowded else "B1"] = {"kl_per_position": kl / npos, "top1_agree_all_positions": agree_all / npos,
                                            "top1_agree_last_position": agree_last / len(ords)}
    out["utility_vs_original_tail"] = util
    print("utility", util, flush=True)

    # ---- all-position k-ULP jitter after k/v/down_proj in layers 0-35, fresh draws per request
    for ki, k in enumerate((1, 2, 4)):
        jit.gen = torch.Generator(device="cuda").manual_seed(SEED * 7 + ki)
        rows = []
        for rec in test:
            pos, neg = pid(rec)
            if regime == "kv_pinned":
                jit.k = k
                l1 = full_logits([rec["input_ids"]])
                l32 = full_logits([rec["input_ids"]] + companions_test(ftest, rec, CROWD))
            elif regime in SOLO_GEOM:
                hs = {}
                for B in (1, CROWD):
                    cap = L34(model)
                    jit.k = k
                    full_logits([rec["input_ids"]] + (companions_test(ftest, rec, B) if B > 1 else []))
                    cap.remove()
                    hs[B] = cap.value[0]
                s, c = transform(hs[1], hs[CROWD])
                l1, l32 = replay(s, False), replay(c, True)
            else:
                jit.k = k
                hs, _ = plain[rec["id"]]
                l1, l32 = replay(hs, False), replay(hs, True)
            jit.k = 0
            rows.append({"id": rec["id"], "task": rec["task"], "B1": answer(l1, pos, neg), "B32": answer(l32, pos, neg)})
        out["jitter"][str(k)] = rows
        print("jitter", k, "done", flush=True)

    # ---- circuit audit: position / head / component patching, replay in [1, L] geometry
    if regime in ("body_only", "kv_pinned"):
        src = kvt if regime == "kv_pinned" else plain
        HEAD = 128
        st = {"patch": None, "record": False, "store": {}}

        def o_pre(li):
            def hook(m, a):
                x = a[0]
                if st["record"]:
                    st["store"][("o", li)] = x.detach().clone()
                p = st["patch"]
                if p and p[0] == "head" and p[1] == li:
                    x = x.clone()
                    sl = slice(p[2] * HEAD, (p[2] + 1) * HEAD)
                    x[:, :, sl] = st["store"][("o", li)][:, :, sl]
                    return (x,)
            return hook

        def comp_hook(kind, li):
            def hook(m, a, o):
                y = o[0] if isinstance(o, tuple) else o
                if st["record"]:
                    st["store"][(kind, li)] = y.detach().clone()
                p = st["patch"]
                if p and p[0] == kind and p[1] == li:
                    y = st["store"][(kind, li)]
                    return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            return hook

        hh = []
        for li in (TAIL_START, TAIL_START + 1):
            hh.append(layers[li].self_attn.o_proj.register_forward_pre_hook(o_pre(li)))
            hh.append(layers[li].self_attn.register_forward_hook(comp_hook("attn", li)))
            hh.append(layers[li].mlp.register_forward_hook(comp_hook("mlp", li)))
        prows = []
        for rec in test:
            pos, neg = pid(rec)
            hs, hc = transform(*src[rec["id"]])
            L = hs.shape[0]

            def lo(h, crowded=False):
                return answer(replay(h, crowded), pos, neg)["log_odds"]

            row = {"id": rec["id"], "task": rec["task"], "L": L,
                   "factorial": {"solo_in_solo": lo(hs), "crowd_in_solo": lo(hc),
                                 "solo_in_crowd": lo(hs, True), "crowd_in_crowd": lo(hc, True)},
                   "differs": (hs != hc).any(-1).tolist(), "single": {}, "group": {}}
            for p in range(L):
                if not row["differs"][p]:
                    row["single"][p] = row["factorial"]["solo_in_solo"]
                    continue
                h = hs.clone()
                h[p] = hc[p]
                row["single"][p] = lo(h)
            body = list(range(3, L - 9))
            q = [body[i * len(body) // 5:(i + 1) * len(body) // 5] for i in range(5)]
            groups = {"prefix_0_2": [0, 1, 2], "body_all": body, "suffix_all": list(range(L - 9, L)),
                      **{f"body_q{i + 1}": g for i, g in enumerate(q)}, "body_last_token": [L - 10]}
            for gname, idx in groups.items():
                h = hs.clone()
                h[idx] = hc[idx]
                row["group"][gname] = lo(h)
            st.update(record=True, store={})
            with torch.no_grad():
                tail_forward(model, hc.unsqueeze(0))
            st["record"] = False
            heads = {}
            for li in (TAIL_START, TAIL_START + 1):
                vals = []
                for hd in range(32):
                    st["patch"] = ("head", li, hd)
                    vals.append(lo(hs))
                heads[str(li)] = vals
            comps = {}
            for kind in ("attn", "mlp"):
                for li in (TAIL_START, TAIL_START + 1):
                    st["patch"] = (kind, li)
                    comps[f"{kind}{li}"] = lo(hs)
            st["patch"] = None
            row["heads"], row["components"] = heads, comps
            prows.append(row)
        for h in hh:
            h.remove()
        out["patching"] = prows
        print("patching done", flush=True)
    write_json(f"{RES}/phase7_eval_{tag}.json", out)
    volume.commit()
    s = {}
    for name, rows in out["test"].items():
        for task in ("detection", "animal"):
            rs = [r for r in rows if r["task"] == task]
            d = [r["B32"]["log_odds"] - r["B1"]["log_odds"] for r in rs]
            s[f"{name}_{task}"] = {"correct": sum(r["B1"]["top_is_neg"] + r["B32"]["top_is_pos"] for r in rs),
                                   "mean_delta": sum(d) / len(d)}
    return s


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3600, memory=32768)
def leak_check(a_tag: str = "body_only_scaled_lr3e-5"):
    """Is Regime A's 'body' signal the canary leaking into body tokens through attention to positions 0-2?
    (1) Zero-training test: the trained A tail on test inputs computed with the prefix K/V pinned in
        layers 0-33 (template positions still set to their solo values, both runs at [1, L]).
    (2) Cache statistics: per prompt, the mean body difference vector (crowd - solo) at the layer-34
        input; cosine similarity between prompts of the same length L (same cuBLAS regime) vs different
        L, without and with the prefix K/V pinned."""
    import itertools
    import random

    import torch

    from wcrowd_runtime import answer, install, load_ckpt, load_model, template_idx
    from wcrowd_tail import tail_forward

    model, _ = load_model()
    ans = json.load(open("/vol/data/bank.json"))["answer_ids"]
    out = {"a_tag": a_tag}
    plain, pmeta, _ = load_acts(["plain_test"])
    kvp, _, _ = load_acts(["kvpin_test"])
    for name, state in (("adaptive_A", load_ckpt(f"{CK}/adaptive_{a_tag}.pt")),
                        ("original_scaled", load_ckpt(f"{CK}/native_tail_scaled.pt"))):
        install(model, state)
        res = {}
        for src_name, src in (("plain_inputs", plain), ("prefix_kv_pinned_inputs", kvp)):
            rows = []
            for key, (hs, hc) in src.items():
                t = pmeta[key]["task"]
                pos, neg = (ans["Yes"], ans["No"]) if t == "detection" else (ans["owl"], ans["cat"])
                hc2 = hc.clone()
                idx = template_idx(hs.shape[0])
                hc2[idx] = hs[idx]
                with torch.no_grad():
                    l1 = tail_forward(model, hs.unsqueeze(0))[0, -1].float()
                    l32 = tail_forward(model, hc2.unsqueeze(0))[0, -1].float()
                a, b = answer(l1, pos, neg), answer(l32, pos, neg)
                rows.append({"task": t, "delta": b["log_odds"] - a["log_odds"],
                             "correct": int(a["top_is_neg"]) + int(b["top_is_pos"])})
            for t in ("detection", "animal"):
                rs = [r for r in rows if r["task"] == t]
                res[f"{src_name}_{t}"] = {"mean_delta": sum(r["delta"] for r in rs) / len(rs),
                                          "frac_delta_positive": sum(r["delta"] > 0 for r in rs) / len(rs),
                                          "correct": sum(r["correct"] for r in rs), "n": len(rs)}
        out[name] = res
        print(name, json.dumps(res), flush=True)

    def body_dirs(name):
        acts, meta, _ = load_acts([name])
        dirs = {}
        for key, (hs, hc) in acts.items():
            L = hs.shape[0]
            d = (hc[3:L - 9].float() - hs[3:L - 9].float()).mean(0)
            dirs[key] = (L, d / d.norm().clamp_min(1e-30))
        return dirs

    rng = random.Random(SEED)
    for name in ("train", "kvpin_train"):
        dirs = body_dirs(name)
        by_L = {}
        for k, (L, d) in dirs.items():
            by_L.setdefault(L, []).append(d)
        same, diff = [], []
        for L, ds in by_L.items():
            for a, b in itertools.combinations(ds[:12], 2):
                same.append(float(a @ b))
        keys = list(dirs)
        for _ in range(2000):
            a, b = rng.sample(keys, 2)
            if dirs[a][0] != dirs[b][0]:
                diff.append(float(dirs[a][1] @ dirs[b][1]))
        out[f"body_direction_cosine_{name}"] = {"same_L_mean": sum(same) / len(same), "n_same": len(same),
                                                "different_L_mean": sum(diff) / len(diff), "n_diff": len(diff)}
        print(name, out[f"body_direction_cosine_{name}"], flush=True)
    write_json(f"{RES}/phase7_leak_check.json", out)
    volume.commit()
    return out


@app.function(image=hf_image, volumes=VOLUMES, timeout=3600, cpu=4, memory=16384)
def analyze7():
    from wcrowd_phase7_analysis import run

    out = run(RES)
    volume.commit()
    return out


# ----------------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------------
LEAN = [("kv_pinned", "scaled", 5e-6, ""), ("kv_pinned", "scaled", 3e-5, "_lr3e-5"),
        ("kv_pinned", "spec", 3e-5, "_lr3e-5"), ("body_only", "scaled", 5e-6, ""),
        ("body_only", "scaled", 3e-5, "_lr3e-5"), ("tail_only", "spec", 3e-5, "_lr3e-5"),
        ("canary_only", "spec", 5e-6, "")]


@app.local_entrypoint()
def lean():
    """Finish Phase 7 after the spend-limit stop: all runs in parallel, each evaluated as it ends."""
    trains = {f"{r}_{s}{v}": (r, s, v, train.spawn(r, s, lr=lr, variant=v)) for r, s, lr, v in LEAN}
    evals = {}
    for name, (r, s, v, call) in trains.items():
        try:
            print("train", name, json.dumps(call.get())[:300], flush=True)
            evals[name] = final_eval.spawn(r, s, variant=v)
        except Exception as exc:  # noqa: BLE001
            print("train", name, "FAILED", repr(exc)[:800], flush=True)
    for name, call in evals.items():
        try:
            print("eval", name, json.dumps(call.get())[:1200], flush=True)
        except Exception as exc:  # noqa: BLE001
            print("eval", name, "FAILED", repr(exc)[:800], flush=True)
    print("analysis", json.dumps(analyze7.remote())[:2000], flush=True)


@app.local_entrypoint()
def run_variant(regime: str, scale: str, lr: float, variant: str):
    """Train and evaluate one extra configuration (e.g. a higher learning rate)."""
    print("train", json.dumps(train.remote(regime, scale, lr=lr, variant=variant))[:400])
    print("eval", json.dumps(final_eval.remote(regime, scale, variant=variant))[:1500])


@app.local_entrypoint()
def extras():
    """Positive control (canary_only) and a 6x learning-rate stress test of body_only."""
    specs = [("canary_only", "spec", 5e-6, ""), ("body_only", "scaled", 3e-5, "_lr3e-5")]
    trains = {f"{r}_{s}{v}": (r, s, v, train.spawn(r, s, lr=lr, variant=v)) for r, s, lr, v in specs}
    evals = {}
    for name, (r, s, v, call) in trains.items():
        print("train", name, json.dumps(call.get())[:400])
        evals[name] = final_eval.spawn(r, s, variant=v)
    for name, call in evals.items():
        print("eval", name, json.dumps(call.get())[:1500])


@app.local_entrypoint()
def main(smoke: bool = False, skip_cache: bool = False, only: str = ""):
    sfx = "_smoke" if smoke else ""
    runs = [("body_only", "spec"), ("body_only", "scaled"), ("tail_only", "spec"), ("tail_only", "scaled"),
            ("kv_pinned", "spec"), ("kv_pinned", "scaled")]
    if only:
        runs = [r for r in runs if f"{r[0]}_{r[1]}" in only.split(",")]
    if smoke:
        runs = [("body_only", "spec"), ("tail_only", "spec"), ("kv_pinned", "spec")]
    # A and B need no new training caches: start them right away
    trains = {f"{r}_{s}": train.spawn(r, s, smoke=smoke) for r, s in runs if r != "kv_pinned"}
    if not skip_cache:
        jobs = [("plain", "test", 0, 1), ("kvpin", "test", 0, 1), ("kvpin", "train", 0, 1), ("kvpin", "val", 0, 1)]
        if not smoke and any(r == "kv_pinned" and s == "scaled" for r, s in runs):
            jobs += [("kvpin", "xtrain", k, 4) for k in range(4)]
        for res in cache.starmap([(a, b, c, d, smoke) for a, b, c, d in jobs]):
            print("cache", json.dumps(res))
    trains.update({f"{r}_{s}": train.spawn(r, s, smoke=smoke) for r, s in runs if r == "kv_pinned"})
    evals = {}
    for name, call in trains.items():
        try:
            print("train", name, json.dumps(call.get())[:400])
            r, s = name.rsplit("_", 1)
            evals[name] = final_eval.spawn(r, s, smoke=smoke)
        except Exception as exc:  # noqa: BLE001
            print("train", name, "FAILED", repr(exc)[:1500])
    for name, call in evals.items():
        try:
            print("eval", name, json.dumps(call.get())[:1500])
        except Exception as exc:  # noqa: BLE001
            print("eval", name, "FAILED", repr(exc)[:1500])
    if not smoke:
        print(json.dumps(analyze7.remote())[:3000])
