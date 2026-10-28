"""Phase 6: M = B x L controls and inference-time defenses (no retraining).

Part 1 (controls)
  prep          exact cuBLAS kernel map at every integer token count M (the Phase 4 map sampled
                every 16 above M = 128), and the bank of position-2 layer-34 bit patterns seen in
                training (train + val + xtrain caches).
  factor_bits   1A: is the layer-34 state at positions 0-2 a function of M = B*L alone? Every
                factorization of M = 1024, 1536, 3072, with per-op attribution of the first
                differing op and the SDPA kernels each factorization launches.
  sweep         1B: B = 1 throughout, prompts lengthened with neutral text to L tokens across the
                split-K boundaries; base / spec / scaled tails; position-2 pattern lookup. Also the
                M = 3072 factorizations with every test prompt, to read the detector there.
  decode_audit  1C: one decode step (L_step = 1, so M = B) after prefill at L = 48 and 1024;
                prefill {B=1, B=32} x decode {B=1, B=32}, with op isolation of the decode step.
Part 2 (defenses, scaled checkpoint)
  flags         2A: allow_bf16_reduced_precision_reduction=False; deterministic algorithms with
                CUBLAS_WORKSPACE_CONFIG :4096:8 and :16:8 (one container each: the workspace size is
                read once per process).
  defenses      2B: canary pinning at the layer-34 input (exact tail replay) and as a prefix-cache
                simulation (K/V of positions 0-2 pinned in all 36 layers). 2C: stochastic k-ULP
                jitter at positions 0-2 or after k/v/down_proj everywhere, with utility cost on
                ordinary prompts.
  analyze6      merges everything into phase6_controls_defenses.json and draws fig5 / fig6.

Inputs: native_tail_{main,scaled}.pt, the bank, the test fillers, the Phase 2 caches. No weights
change. Outputs: /vol/results/phase6_*.json, merged into phase6_controls_defenses.json, and
fig5/fig6 (all mirrored in results_watching_crowd/).

    modal run 06_controls_and_defenses.py::main [--smoke]
    python3 make_phase6_tables.py sweep                   # Section 7-8 tables from the merged JSON
"""

import json
import os
import random

import modal

from wcrowd_common import (
    GPU, MODEL_ID, OVERHEAD, PREFIX_IDS, REVISION, SEED, SUFFIX_IDS, TAIL_START, VOLUMES,
    env_info, hf_image, volume, write_json,
)

app = modal.App("watching-crowd-06-controls")
CROWD = 32
SWEEP_L = [64, 128, 200, 256, 260, 384, 512, 830, 832, 960, 1024, 1300, 1536]
FACTOR_M = {
    1024: [(32, 32), (16, 64), (8, 128), (4, 256), (2, 512), (1, 1024)],
    1536: [(32, 48), (16, 96), (8, 192), (4, 384), (2, 768), (1, 1536)],
    3072: [(32, 96), (16, 192), (8, 384), (4, 768), (2, 1536), (1, 3072)],
}
TEMPLATE_PREFIX = [151644, 872, 198]  # <|im_start|>user\n
SEP = 271  # "\n\n": closes the neutral padding block before the original user content
RES = "/vol/results"


# ----------------------------------------------------------------------------------------------
# shared helpers (run inside containers)
# ----------------------------------------------------------------------------------------------
class Ctx:
    """Model + tokenizer + bank + the three tails (base/main/scaled) kept on GPU for fast swaps."""

    def __init__(self, tails=("base", "main", "scaled")):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="cuda",
        ).eval()
        self.model.requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        keep = (f"model.layers.{TAIL_START}.", f"model.layers.{TAIL_START + 1}.", "model.norm.")
        self.named = {n: p for n, p in self.model.named_parameters() if n.startswith(keep)}
        self.tails = {}
        for tag in tails:
            if tag == "base":
                self.tails[tag] = {n: p.detach().clone() for n, p in self.named.items()}
            else:
                sd = torch.load(f"/vol/checkpoints/native_tail_{tag}.pt", map_location="cpu")["state_dict"]
                self.tails[tag] = {n: w.to("cuda") for n, w in sd.items()}
        self.current = "base"
        bank = json.load(open("/vol/data/bank.json"))
        self.answers = bank["answer_ids"]
        self.test = [r for r in bank["records"] if r["split"] == "test" and r["task"] in ("detection", "animal")]
        self.ordinary = [r for r in bank["records"] if r["split"] == "val" and r["task"] == "ordinary"]
        fillers = json.load(open("/vol/data/test_fillers.json"))["fillers"]
        self.ftok = [self.tok.encode(f["text"], add_special_tokens=False) for f in fillers]
        stream = []
        for t in self.ftok:
            stream += t + [SEP]
        self.stream = stream

    def use(self, tag):
        with self.torch.no_grad():
            for n, w in self.tails[tag].items():
                self.named[n].copy_(w)
        self.current = tag

    # -- prompt construction --------------------------------------------------------------
    def extend(self, rec, L, salt=0):
        """The prompt at exactly L tokens: neutral test-filler text (cut to fit, then "\\n\\n")
        inserted right after <|im_start|>user\\n, before the original user content."""
        n = L - rec["L"]
        if n < 0:
            return None
        if n == 0:
            return list(rec["input_ids"])
        content = rec["input_ids"][len(PREFIX_IDS):-len(SUFFIX_IDS)]
        off = (hash_id(rec["id"]) * 7919 + salt * 104729) % (len(self.stream) - 5000)
        return PREFIX_IDS + self.stream[off: off + n - 1] + [SEP] + content + SUFFIX_IDS

    def neutral(self, L, j):
        """A neutral companion sequence of exactly L tokens (j selects the text)."""
        body = L - OVERHEAD
        if body <= len(self.ftok[j % len(self.ftok)]):
            return PREFIX_IDS + self.ftok[j % len(self.ftok)][:body] + SUFFIX_IDS
        off = (j * 15485863) % (len(self.stream) - body - 1)
        return PREFIX_IDS + self.stream[off: off + body] + SUFFIX_IDS

    def companions(self, rec, B, L=None):
        """Phase 4's companions: the same seeded test fillers, cut to the target's length."""
        L = L or rec["L"]
        rng = random.Random(f"test:{SEED}:{rec['id']}")
        idx = rng.sample(range(len(self.ftok)), 63)[: B - 1]
        return [PREFIX_IDS + self.ftok[i][: L - OVERHEAD] + SUFFIX_IDS for i in idx]

    # -- forward passes --------------------------------------------------------------------
    def logits(self, seqs, keep=1):
        ids = self.torch.tensor(seqs, dtype=self.torch.long, device="cuda")
        with self.torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=None, use_cache=False, logits_to_keep=keep)
        return out.logits

    def answer(self, logits_row, rec):
        lp = logits_row.double().log_softmax(-1)
        pos, neg = self.answers[rec["pos"]], self.answers[rec["neg"]]
        top = int(logits_row.argmax())
        return {"log_odds": float(lp[pos] - lp[neg]), "top": top,
                "top_is_pos": top == pos, "top_is_neg": top == neg}


def hash_id(s):
    import hashlib

    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def vec_hash(t):
    import hashlib

    import torch

    return hashlib.sha1(t.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


class L34:
    """Read-only pre-hook on layer 34: rows `rows`, all positions (eager mode: cannot change bits)."""

    def __init__(self, model, rows=(0,)):
        self.rows, self.value = list(rows), None
        self.h = model.model.layers[TAIL_START].register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.value = args[0][self.rows].detach().clone()

    def remove(self):
        self.h.remove()


class OpRecorder:
    """Read-only hooks on every layer: q/k/v_proj outputs, attention output (o_proj input),
    down_proj output and layer output, for row `row` at `positions`, in execution order."""

    def __init__(self, model, positions, row=0):
        self.store, self.handles, self.order = {}, [], []
        for li, layer in enumerate(model.model.layers):
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                self._out(layer.get_submodule(name), f"{li}.{name.split('.')[-1]}", positions, row)
            self._pre(layer.self_attn.o_proj, f"{li}.attn_out", positions, row)
            self._out(layer.mlp.down_proj, f"{li}.down_proj", positions, row)
            self._out(layer, f"{li}.layer_out", positions, row)

    def _out(self, mod, key, positions, row):
        self.order.append(key)

        def hook(m, a, o):
            y = o[0] if isinstance(o, tuple) else o
            self.store[key] = y[row, positions].detach().clone()
        self.handles.append(mod.register_forward_hook(hook))

    def _pre(self, mod, key, positions, row):
        self.order.append(key)

        def hook(m, a):
            self.store[key] = a[0][row, positions].detach().clone()
        self.handles.append(mod.register_forward_pre_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def first_difference(order, a, b):
    import torch

    for key in order:
        if not torch.equal(a[key], b[key]):
            d = (a[key].float() - b[key].float())
            return {"op": key, "linf": float(d.abs().max()), "frac_differ": float((d != 0).float().mean())}
    return None


def cuda_kernels(fn):
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def kernel_summary(names):
    from collections import Counter

    c = Counter(names)
    return {
        "splitKreduce": sum(v for k, v in c.items() if "splitKreduce" in k),
        "splitKreduce_templates": sorted({k.split("(")[0][:160] for k in c if "splitKreduce" in k}),
        "gemv": sum(v for k, v in c.items() if "gemv" in k),
        "flash": {k.split("(")[0][:140]: v for k, v in c.items() if "flash" in k.lower()},
        "attention_other": {k.split("(")[0][:140]: v for k, v in c.items()
                            if ("fmha" in k.lower() or "attention" in k.lower()) and "flash" not in k.lower()},
        "total_kernels": sum(c.values()),
    }


def diff_stats(a, b):
    a, b = a.float(), b.float()
    d = a - b
    return {"bit_identical": bool((d == 0).all()), "linf": float(d.abs().max()),
            "rel_l2": float(d.norm() / a.norm().clamp_min(1e-30)),
            "frac_differ": float((d != 0).float().mean())}


# ----------------------------------------------------------------------------------------------
# prep: exact kernel map + position-2 pattern bank
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, memory=32768)
def prep(smoke: bool = False):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file

    shapes = {"q_proj/o_proj": (4096, 4096), "k_proj/v_proj": (1024, 4096),
              "gate_proj/up_proj": (12288, 4096), "down_proj": (4096, 12288)}
    ms = list(range(1, 1601)) + list(range(1608, 3201, 8)) if not smoke else list(range(250, 270))
    g = torch.Generator(device="cuda").manual_seed(SEED)
    kmap = {}
    from torch.profiler import ProfilerActivity, profile

    marker = torch.zeros(7, device="cuda")
    for name, (n, k) in shapes.items():
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        x_all = torch.randn(max(ms), k, device="cuda", generator=g).to(torch.bfloat16)
        rows = []
        for c0 in range(0, len(ms), 100):
            chunk = ms[c0: c0 + 100]
            for m in chunk:
                F.linear(x_all[:m], w)  # warm-up: heuristics cached, nothing lazily initialized
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for m in chunk:
                    F.linear(x_all[:m], w)
                    marker.fill_(1.0)  # FillFunctor kernel separates one GEMM call from the next
                torch.cuda.synchronize()
            evs = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                         key=lambda e: e.time_range.start)
            groups, cur = [], []
            for e in evs:
                if "FillFunctor" in e.name:
                    groups.append(cur)
                    cur = []
                else:
                    cur.append(e.name)
            assert len(groups) == len(chunk), (name, len(groups), len(chunk))
            for m, names in zip(chunk, groups):
                main_k = [s_.split("(")[0][:120] for s_ in names if "splitKreduce" not in s_ and "Memset" not in s_]
                rows.append((m, any("splitKreduce" in s_ for s_ in names), "|".join(main_k)))
        regimes = []
        for m, sk, kern in rows:
            if regimes and regimes[-1]["split_k"] == sk and regimes[-1]["kernel"] == kern:
                regimes[-1]["M_end"] = m
            else:
                regimes.append({"M_start": m, "M_end": m, "split_k": sk, "kernel": kern})
        sk_ranges = []
        for m, sk, _ in rows:
            if sk and sk_ranges and sk_ranges[-1][1] == m - (1 if m <= 1600 else 8):
                sk_ranges[-1][1] = m
            elif sk:
                sk_ranges.append([m, m])
        kmap[name] = {"N": n, "K": k, "regimes": regimes, "split_k_ranges": sk_ranges,
                      "probed_M": "every integer 1..1600, every 8 in 1608..3200"}
        print(name, "split-K ranges", sk_ranges, flush=True)

    # position-2 layer-34 patterns seen in training (Phase 2 caches)
    bank = {"solo": {}, "crowd": {}}
    tensors = {}
    for split in ("train", "val", "xtrain"):
        path = f"/vol/activations/{split}.safetensors"
        if not os.path.exists(path):
            continue
        rows = [json.loads(line) for line in open(f"/vol/activations/{split}_rows.jsonl")]
        with safe_open(path, "pt") as f:
            lengths = f.get_tensor("lengths").tolist()
            starts = [0]
            for L in lengths:
                starts.append(starts[-1] + L)
            for cond, mult in (("solo", 1), ("crowd", CROWD)):
                sl = f.get_slice(cond)
                for i, L in enumerate(lengths[: (50 if smoke else len(lengths))]):
                    v = sl[starts[i] + 2: starts[i] + 3]
                    h = vec_hash(v)
                    e = bank[cond].setdefault(h, {"M": set(), "L": set(), "splits": set(), "n": 0})
                    if f"{cond}_{h}" not in tensors:
                        tensors[f"{cond}_{h}"] = v[0].clone()
                    e["M"].add(L * mult)
                    e["L"].add(L)
                    e["splits"].add(split)
                    e["n"] += 1
        print(split, {c: len(bank[c]) for c in bank}, flush=True)
    bank_json = {c: {h: {"M": sorted(e["M"]), "L": sorted(e["L"]), "splits": sorted(e["splits"]), "n": e["n"]}
                     for h, e in d.items()} for c, d in bank.items()}
    for c, d in bank_json.items():
        for k_, (h, e) in enumerate(sorted(d.items(), key=lambda kv: min(kv[1]["M"]))):
            e["label"] = f"{c}_{k_}"

    # reference position-2 vector at [B=32, L=32] (M = 1024) for Test 1B
    model_ctx = Ctx(tails=("base",))
    cap = L34(model_ctx.model)
    model_ctx.logits([model_ctx.neutral(32, j) for j in range(CROWD)])
    cap.remove()
    tensors["ref_32x32_pos012"] = cap.value[0, :3].clone()
    save_file({k: v.contiguous() for k, v in tensors.items()}, f"{RES}/phase6_pos2_bank.safetensors")
    out = {"kernel_map_exact": kmap, "pos2_bank": bank_json, "env": env_info()}
    write_json(f"{RES}/phase6_prep.json", out)
    volume.commit()
    return {"split_k_ranges": {k: v["split_k_ranges"] for k, v in kmap.items()},
            "bank_sizes": {c: len(d) for c, d in bank_json.items()}}


def load_bank():
    from safetensors.torch import load_file

    meta = json.load(open(f"{RES}/phase6_prep.json"))["pos2_bank"]
    vecs = load_file(f"{RES}/phase6_pos2_bank.safetensors")
    lookup = {}
    for cond, d in meta.items():
        for h, e in d.items():
            lookup[h] = {"cond": cond, "label": e["label"], "M": e["M"]}
    return lookup, vecs["ref_32x32_pos012"].cuda()


# ----------------------------------------------------------------------------------------------
# 1A: factorizations of M
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600)
def factor_bits(smoke: bool = False):
    import torch

    ctx = Ctx(tails=("base", "scaled"))
    model = ctx.model
    short = sorted(ctx.test, key=lambda r: r["L"])
    picks = [r for r in short if r["task"] == "detection"][:4] + [r for r in short if r["task"] == "animal"][:4]
    if smoke:
        picks = picks[:1]
    results = {"prompts": [r["id"] for r in picks], "prompt_L": [r["L"] for r in picks], "M": {}}
    for M, facs in FACTOR_M.items():
        if smoke and M != 1024:
            continue
        res = {"factorizations": [list(f) for f in facs], "per_prompt": [], "kernels": {}}
        for pi, rec in enumerate(picks):
            states, preds = {}, {}
            for B, L in facs:
                row0 = ctx.extend(rec, L)
                fits = row0 is not None
                if not fits:
                    row0 = ctx.neutral(L, 999)
                seqs = [row0] + [ctx.neutral(L, j) for j in range(1, B)]
                rec_ops = OpRecorder(model, [0, 1, 2])
                cap = L34(model)
                lg = ctx.logits(seqs)
                cap.remove()
                rec_ops.remove()
                states[(B, L)] = {"ops": dict(rec_ops.store), "h34": cap.value[0, :3].clone()}
                pr = {"prompt_fits": fits}
                if fits:
                    pr["base"] = ctx.answer(lg[0, -1].float(), rec)
                    ctx.use("scaled")
                    pr["scaled"] = ctx.answer(ctx.logits(seqs)[0, -1].float(), rec)
                    ctx.use("base")
                preds[f"{B}x{L}"] = pr
                if pi == 0:
                    res["kernels"][f"{B}x{L}"] = kernel_summary(cuda_kernels(lambda: ctx.logits(seqs)))
            ref = states[(1, M)]
            order = rec_ops.order
            entry = {"id": rec["id"], "vs_long_solo": {}, "pairwise_h34_identical": {}, "predictions": preds}
            for B, L in facs:
                s = states[(B, L)]
                entry["vs_long_solo"][f"{B}x{L}"] = {
                    "h34_pos012": diff_stats(ref["h34"], s["h34"]),
                    "h34_pos2": diff_stats(ref["h34"][2], s["h34"][2]),
                    "first_differing_op": first_difference(order, ref["ops"], s["ops"]),
                    "h34_pos2_hash": vec_hash(s["h34"][2]),
                }
            keys = [f"{B}x{L}" for B, L in facs]
            for i, (B1, L1) in enumerate(facs):
                for B2, L2 in facs[i + 1:]:
                    entry["pairwise_h34_identical"][f"{B1}x{L1}|{B2}x{L2}"] = bool(
                        torch.equal(states[(B1, L1)]["h34"], states[(B2, L2)]["h34"]))
            entry["keys"] = keys
            res["per_prompt"].append(entry)
            print(M, rec["id"], {k: v["h34_pos012"]["bit_identical"] for k, v in entry["vs_long_solo"].items()},
                  {k: (v["first_differing_op"] or {}).get("op") for k, v in entry["vs_long_solo"].items()}, flush=True)
        results["M"][str(M)] = res
    results["env"] = env_info()
    write_json(f"{RES}/phase6_factor_bits{'_smoke' if smoke else ''}.json", results)
    volume.commit()
    return {M: [{k: v["h34_pos012"]["bit_identical"] for k, v in e["vs_long_solo"].items()}
                for e in r["per_prompt"]] for M, r in results["M"].items()}


# ----------------------------------------------------------------------------------------------
# 1B: solo length sweep (+ M = 3072 factorization predictions), sharded over test prompts
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600, memory=32768)
def sweep(shard: int = 0, n_shards: int = 1, lengths: list | None = None, smoke: bool = False):
    import torch

    ctx = Ctx()
    lookup, ref1024 = load_bank()
    lengths = sorted(set(lengths or SWEEP_L))
    recs = ctx.test[shard::n_shards]
    if smoke:
        recs, lengths = recs[:2], lengths[:3]
    rows = {r["id"]: {"id": r["id"], "task": r["task"], "L_natural": r["L"],
                      "family_seen_in_training": r["family_seen_in_training"], "solo": {}, "f3072": {}}
            for r in recs}
    for tag in ("base", "main", "scaled"):
        ctx.use(tag)
        for rec in recs:
            row = rows[rec["id"]]
            for L in [rec["L"]] + [x for x in lengths if x > rec["L"]]:
                ids = ctx.extend(rec, L)
                cap = L34(ctx.model) if tag == "base" else None
                lg = ctx.logits([ids])
                e = row["solo"].setdefault(str(L), {})
                e[tag] = ctx.answer(lg[0, -1].float(), rec)
                if cap is not None:
                    cap.remove()
                    p2 = cap.value[0, 2]
                    h = vec_hash(p2)
                    e["pos2_hash"] = h
                    e["pos2_bank"] = lookup.get(h, {"cond": "new", "label": "new", "M": []})
                    e["pos2_frac_differ_vs_32x32"] = float((p2 != ref1024[2]).float().mean())
                    e["pos012_identical_to_32x32"] = bool(torch.equal(cap.value[0, :3], ref1024))
            if smoke:
                continue
            for B, L in FACTOR_M[3072]:
                seqs = [ctx.extend(rec, L)] + [ctx.neutral(L, j) for j in range(1, B)]
                cap = L34(ctx.model) if tag == "base" else None
                lg = ctx.logits(seqs)
                e = row["f3072"].setdefault(f"{B}x{L}", {})
                e[tag] = ctx.answer(lg[0, -1].float(), rec)
                if cap is not None:
                    cap.remove()
                    e["pos012_hash"] = vec_hash(cap.value[0, :3])
                    e["pos2_bank"] = lookup.get(vec_hash(cap.value[0, 2]), {"cond": "new", "label": "new", "M": []})
        print(tag, "shard", shard, "done", flush=True)
    return list(rows.values())


@app.function(image=hf_image, volumes=VOLUMES, timeout=600)
def save_json(name: str, obj):
    write_json(f"{RES}/{name}", obj)
    volume.commit()
    return name


# ----------------------------------------------------------------------------------------------
# 1C: decode-step audit
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, memory=32768)
def decode_audit(smoke: bool = False):
    import copy

    import torch
    import torch.nn.functional as F

    ctx = Ctx(tails=("base",))
    model = ctx.model
    short = sorted(ctx.test, key=lambda r: r["L"])
    picks = [r for r in short if r["task"] == "detection"][:4] + [r for r in short if r["task"] == "animal"][:4]
    if smoke:
        picks = picks[:1]

    def prefill(seqs):
        ids = torch.tensor(seqs, device="cuda")
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=None, use_cache=True, logits_to_keep=1)
        return out.past_key_values, int(out.logits[0, -1].argmax())

    def decode(cache, tok, B, record=True):
        c = copy.deepcopy(cache)
        if c.layers[0].keys.shape[0] == 1 and B > 1:
            c.batch_repeat_interleave(B)
        ids = torch.full((B, 1), tok, device="cuda")
        rec_ops = OpRecorder(model, [0]) if record else None
        cap = L34(model)
        with torch.no_grad():
            lg = model(input_ids=ids, past_key_values=c, attention_mask=None, use_cache=True,
                       logits_to_keep=1).logits[0, -1].float()
        cap.remove()
        if rec_ops:
            rec_ops.remove()
        return {"h34": cap.value[0, 0].clone(), "ops": dict(rec_ops.store) if rec_ops else None,
                "logits": lg, "order": rec_ops.order if rec_ops else None}

    out = {"prompts": [r["id"] for r in picks], "L": {}}
    for L in (48, 1024):
        per = []
        for pi, rec in enumerate(picks):
            target = ctx.extend(rec, L)
            comp = [ctx.neutral(L, j) for j in range(1, CROWD)]
            cache1, tok = prefill([target])
            cache32, tok32 = prefill([target] + comp)
            row0 = copy.deepcopy(cache32)
            row0.batch_select_indices(torch.tensor([0], device="cuda"))
            # real companions in the decode batch: target's B=1 cache in row 0, fillers' caches after
            mixed = copy.deepcopy(cache32)
            for li, layer in enumerate(mixed.layers):
                layer.keys = torch.cat([cache1.layers[li].keys, cache32.layers[li].keys[1:]], 0)
                layer.values = torch.cat([cache1.layers[li].values, cache32.layers[li].values[1:]], 0)
            runs = {
                "P1_D1": decode(cache1, tok, 1),
                "P1_D32copies": decode(cache1, tok, CROWD),
                "P1_D32real": decode(mixed, tok, CROWD, record=False),
                "P32_D1": decode(row0, tok, 1, record=False),
                "P32_D32": decode(cache32, tok, CROWD, record=False),
            }
            ref = runs["P1_D1"]
            entry = {"id": rec["id"], "first_token": tok, "first_token_B32_prefill": tok32,
                     "prefill_kv_layer34_carryover": diff_stats(cache1.layers[TAIL_START].keys[0],
                                                                 cache32.layers[TAIL_START].keys[0]),
                     "vs_P1_D1": {k: diff_stats(ref["h34"], v["h34"]) for k, v in runs.items() if k != "P1_D1"},
                     "logits_linf_vs_P1_D1": {k: float((ref["logits"] - v["logits"]).abs().max())
                                              for k, v in runs.items() if k != "P1_D1"},
                     "decode_first_differing_op": first_difference(ref["order"], ref["ops"],
                                                                   runs["P1_D32copies"]["ops"]),
                     "copies_equal_real_companions": bool(torch.equal(runs["P1_D32copies"]["h34"],
                                                                      runs["P1_D32real"]["h34"]))}
            if pi == 0:
                # kernels launched by the decode step itself, and op isolation on layer 0 inputs
                c1 = copy.deepcopy(cache1)
                c32 = copy.deepcopy(cache1)
                c32.batch_repeat_interleave(CROWD)
                ids1, ids32 = torch.full((1, 1), tok, device="cuda"), torch.full((CROWD, 1), tok, device="cuda")
                with torch.no_grad():
                    k1 = kernel_summary(cuda_kernels(lambda: model(input_ids=ids1, past_key_values=copy.deepcopy(c1), use_cache=True, logits_to_keep=1)))
                    k32 = kernel_summary(cuda_kernels(lambda: model(input_ids=ids32, past_key_values=copy.deepcopy(c32), use_cache=True, logits_to_keep=1)))
                entry["decode_kernels"] = {"B1": k1, "B32": k32}
                # SDPA isolation: identical q/k/v rows at B=1 vs repeated to B=32
                real = F.scaled_dot_product_attention
                captured = {}

                def wrap(*a, **kw):
                    if "args" not in captured:
                        captured["args"], captured["kw"] = a, kw
                    return real(*a, **kw)
                torch.nn.functional.scaled_dot_product_attention = wrap
                try:
                    with torch.no_grad():
                        model(input_ids=ids1, past_key_values=copy.deepcopy(c1), use_cache=True, logits_to_keep=1)
                finally:
                    torch.nn.functional.scaled_dot_product_attention = real
                a, kw = captured["args"], captured["kw"]
                rep = [x.repeat(CROWD, *([1] * (x.dim() - 1))) if torch.is_tensor(x) and x.dim() == 4 else x for x in a]
                with torch.no_grad():
                    y1 = real(*a, **kw)
                    y32 = real(*rep, **kw)
                entry["sdpa_isolation_layer0"] = diff_stats(y1[0], y32[0])
                entry["sdpa_isolation_shapes"] = [list(x.shape) for x in a if torch.is_tensor(x)]
                entry["sdpa_kernels"] = {"B1": kernel_summary(cuda_kernels(lambda: real(*a, **kw))),
                                         "B32": kernel_summary(cuda_kernels(lambda: real(*rep, **kw)))}
                # GEMM isolation: layer-0 linear modules on the decode token's input row
                lin = {}
                for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                    mod = model.model.layers[0].get_submodule(name)
                    x = torch.randn(1, 1, mod.in_features, device="cuda").to(torch.bfloat16)
                    with torch.no_grad():
                        y1 = mod(x)
                        y32 = mod(x.repeat(CROWD, 1, 1))
                    lin[name] = {**diff_stats(y1[0], y32[0]),
                                 "kernels_B1": kernel_summary(cuda_kernels(lambda: mod(x)))["splitKreduce"],
                                 "kernels_B32": kernel_summary(cuda_kernels(lambda: mod(x.repeat(CROWD, 1, 1))))["splitKreduce"]}
                entry["gemm_isolation_layer0"] = lin
            per.append(entry)
            print(L, rec["id"], {k: v["frac_differ"] for k, v in entry["vs_P1_D1"].items()},
                  entry["decode_first_differing_op"], flush=True)
        out["L"][str(L)] = per
    out["env"] = env_info()
    write_json(f"{RES}/phase6_decode{'_smoke' if smoke else ''}.json", out)
    volume.commit()
    return {L: [{k: round(v["frac_differ"], 3) for k, v in e["vs_P1_D1"].items()} for e in per]
            for L, per in out["L"].items()}


# ----------------------------------------------------------------------------------------------
# 2A: numerical / determinism flags (one container per configuration)
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600, memory=32768,
              single_use_containers=True)
def flags(config: str = "default", smoke: bool = False):
    if config == "det_4096":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif config == "det_16":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import torch
    from safetensors.torch import save_file

    if config.startswith("det"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if config == "no_rpr":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ctx = Ctx(tails=("base", "scaled"))
    recs = ctx.test[:4] if smoke else ctx.test
    gate = [r for r in ctx.test if r["task"] == "detection"][:4] + [r for r in ctx.test if r["task"] == "animal"][:4]
    out = {"config": config, "env": env_info(), "gate": [], "eval": [], "kernels": {}}
    out["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    saved = {}
    for i, rec in enumerate(gate[: (1 if smoke else 8)]):
        hs = {}
        for B in (1, CROWD):
            cap = L34(ctx.model)
            ctx.logits([rec["input_ids"]] + ctx.companions(rec, B))
            cap.remove()
            hs[B] = cap.value[0]
            saved[f"{rec['id']}_B{B}"] = cap.value[0].contiguous()
        out["gate"].append({"id": rec["id"], **diff_stats(hs[1], hs[CROWD])})
        if i == 0:
            for B in (1, CROWD):
                seqs = [rec["input_ids"]] + ctx.companions(rec, B)
                out["kernels"][f"B{B}"] = kernel_summary(cuda_kernels(lambda: ctx.logits(seqs)))
    save_file(saved, f"{RES}/phase6_flags_h34_{config}.safetensors")
    ctx.use("scaled")
    for rec in recs:
        row = {"id": rec["id"], "task": rec["task"], "family_seen_in_training": rec["family_seen_in_training"]}
        for B in (1, CROWD):
            row[f"B{B}"] = ctx.answer(ctx.logits([rec["input_ids"]] + ctx.companions(rec, B))[0, -1].float(), rec)
        out["eval"].append(row)
    # ordinary prompts: last-position log-probs, for KL against the default configuration
    lps = {}
    for rec in ctx.ordinary[: (2 if smoke else 64)]:
        for B in (1, CROWD):
            lg = ctx.logits([rec["input_ids"]] + ctx.companions(rec, B))[0, -1].float()
            lps[f"{rec['id']}_B{B}"] = lg.log_softmax(-1).contiguous()
    save_file(lps, f"{RES}/phase6_flags_ordinary_{config}.safetensors")
    write_json(f"{RES}/phase6_flags_{config}{'_smoke' if smoke else ''}.json", out)
    volume.commit()
    d = [r[f"B{CROWD}"]["log_odds"] - r["B1"]["log_odds"] for r in out["eval"]]
    return {"config": config, "gate_bit_identical": [g["bit_identical"] for g in out["gate"]],
            "gate_frac_differ": [round(g["frac_differ"], 3) for g in out["gate"]],
            "splitK_B1": out["kernels"].get("B1", {}).get("splitKreduce"),
            "splitK_B32": out["kernels"].get(f"B{CROWD}", {}).get("splitKreduce"),
            "mean_delta": sum(d) / len(d)}


# ----------------------------------------------------------------------------------------------
# 2B + 2C: canary pinning and stochastic ULP jamming (scaled checkpoint, default flags)
# ----------------------------------------------------------------------------------------------
def ulp_jitter(x, k, gen):
    """Move every finite nonzero bf16 element by u ULPs, u ~ Uniform{-k..k}; elements whose
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


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=5 * 3600, memory=65536)
def defenses(smoke: bool = False):
    import torch

    from wcrowd_tail import crowd_batch, log_odds, tail_forward

    ctx = Ctx(tails=("base", "scaled"))
    model = ctx.model
    lookup, _ = load_bank()
    recs = ctx.test[:4] if smoke else ctx.test
    ords = ctx.ordinary[: (2 if smoke else 64)]
    out = {"env": env_info()}

    def lo_of(logits_row, rec):
        return ctx.answer(logits_row, rec)

    # ---- capture layer-34 inputs (base prefix; read-only hook) and the fixed-prefix references
    h34 = {}
    for rec in recs:
        for B in (1, CROWD):
            cap = L34(model)
            ctx.logits([rec["input_ids"]] + ctx.companions(rec, B))
            cap.remove()
            h34[(rec["id"], B)] = cap.value[0]
    cap = L34(model)
    ctx.logits([TEMPLATE_PREFIX])
    cap.remove()
    fixed_h34 = cap.value[0, :3].clone()

    # ---- 2B (i): pinning at the layer-34 input, exact tail replay
    ctx.use("scaled")
    levels = {"none": None, "pin2": [2], "pin012": [0, 1, 2], "pin_template": "template",
              "pin012_fixedprefix": "fixed"}
    rows = []
    for rec in recs:
        s, c, L = h34[(rec["id"], 1)], h34[(rec["id"], CROWD)], rec["L"]
        r = {"id": rec["id"], "task": rec["task"], "family_seen_in_training": rec["family_seen_in_training"]}
        for name, lv in levels.items():
            ss, cc = s.clone(), c.clone()
            if lv == "template":
                idx = [0, 1, 2] + list(range(L - 9, L))
                cc[idx] = s[idx]
            elif lv == "fixed":
                ss[:3], cc[:3] = fixed_h34, fixed_h34
            elif lv is not None:
                cc[lv] = s[lv]
            with torch.no_grad():
                l1 = tail_forward(model, ss.unsqueeze(0))[0, -1].float()
                l32 = tail_forward(model, crowd_batch(cc, batch=CROWD))[0, -1].float()
            r[name] = {"B1": lo_of(l1, rec), "B32": lo_of(l32, rec)}
        rows.append(r)
    out["pin_l34"] = rows
    print("pin_l34 done", flush=True)

    # ---- 2B (ii): prefix-cache simulation: K/V of positions P pinned in all 36 layers
    state = {"mode": None, "ref": {}, "pos": None}

    def kv_hook(key):
        def hook(m, a, o):
            if state["mode"] == "record":
                state["ref"][key] = o[0, state["pos"]].detach().clone()
            elif state["mode"] == "pin":
                o = o.clone()
                o[:, state["pos"]] = state["ref"][key].to(o.dtype)
                return o
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(kv_hook(f"{li}.k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(kv_hook(f"{li}.v")))
    kv_rows = []
    fixed_ref = {}
    for pos_name, pos in (("kv2", [2]), ("kv012", [0, 1, 2])):
        state.update(mode="record", ref={}, pos=pos)
        ctx.logits([TEMPLATE_PREFIX])
        fixed_ref[pos_name] = dict(state["ref"])
    for rec in recs:
        r = {"id": rec["id"], "task": rec["task"]}
        for pos_name, pos in (("kv2", [2]), ("kv012", [0, 1, 2])):
            # (a) per-prompt B=1 reference, applied to the B=32 pass (the B=1 pass is its own reference)
            state.update(mode="record", ref={}, pos=pos)
            l1 = ctx.logits([rec["input_ids"]])[0, -1].float()
            state["mode"] = "pin"
            l32 = ctx.logits([rec["input_ids"]] + ctx.companions(rec, CROWD))[0, -1].float()
            r[f"{pos_name}_solo_ref"] = {"B1": lo_of(l1, rec), "B32": lo_of(l32, rec)}
            # (b) one fixed prefix cache (the 3-token prefix computed alone), used by both passes
            state.update(mode="pin", ref=fixed_ref[pos_name], pos=pos)
            l1 = ctx.logits([rec["input_ids"]])[0, -1].float()
            l32 = ctx.logits([rec["input_ids"]] + ctx.companions(rec, CROWD))[0, -1].float()
            r[f"{pos_name}_fixed_ref"] = {"B1": lo_of(l1, rec), "B32": lo_of(l32, rec)}
            state["mode"] = None
        kv_rows.append(r)
    state["mode"] = None
    out["pin_kv"] = kv_rows
    print("pin_kv done", flush=True)

    # ---- ordinary-prompt references (noiseless scaled model), all positions, row 0
    def ord_logprobs(rec, B):
        lg = ctx.logits([rec["input_ids"]] + ctx.companions(rec, B), keep=0)[0].float()
        return lg.log_softmax(-1)

    refs = {(r["id"], B): ord_logprobs(r, B).cpu() for r in ords for B in (1, CROWD)}

    def utility(B):
        kl, top_all, top_last, n_pos = 0.0, 0, 0, 0
        for r in ords:
            ref = refs[(r["id"], B)].cuda()
            mine = ord_logprobs(r, B)
            kl += float((ref.exp() * (ref - mine)).sum(-1).sum())
            top_all += int((ref.argmax(-1) == mine.argmax(-1)).sum())
            top_last += int(ref[-1].argmax() == mine[-1].argmax())
            n_pos += ref.shape[0]
        return {"kl_per_position": kl / n_pos, "top1_agree_all_positions": top_all / n_pos,
                "top1_agree_last_position": top_last / len(ords)}

    # utility of the fixed prefix-cache pinning
    out["pin_kv_utility"] = {}
    for pos_name, pos in (("kv2", [2]), ("kv012", [0, 1, 2])):
        state.update(mode="pin", ref=fixed_ref[pos_name], pos=pos)
        out["pin_kv_utility"][f"{pos_name}_fixed_ref"] = {f"B{B}": utility(B) for B in (1, CROWD)}
        state["mode"] = None
    for h in handles:
        h.remove()

    # ---- 2C: stochastic k-ULP jamming
    jam = {"scope": None, "k": 0, "gen": None}

    def jam_hook(positions_only):
        def hook(m, a, o):
            if jam["scope"] is None:
                return None
            if positions_only and jam["scope"] == "template":
                o = o.clone()
                o[:, :3] = ulp_jitter(o[:, :3], jam["k"], jam["gen"])
                return o
            if not positions_only and jam["scope"] == "all":
                return ulp_jitter(o, jam["k"], jam["gen"])
            return None
        return hook

    handles = []
    for li in range(TAIL_START):
        layer = model.model.layers[li]
        handles.append(layer.mlp.down_proj.register_forward_hook(jam_hook(True)))
        for mod in (layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.down_proj):
            handles.append(mod.register_forward_hook(jam_hook(False)))
    settings = [("none", 0)] + [("template", k) for k in (1, 2, 4, 8)] + [("all", k) for k in (1, 2, 4)]
    if smoke:
        settings = settings[:2] + settings[-1:]
    jam_out = []
    for si, (scope, k) in enumerate(settings):
        jam.update(scope=None if scope == "none" else scope, k=k,
                   gen=torch.Generator(device="cuda").manual_seed(SEED * 1000 + si))
        rows, matches = [], {"B1": [], "B32": []}
        for rec in recs:
            r = {"id": rec["id"], "task": rec["task"], "family_seen_in_training": rec["family_seen_in_training"]}
            for B in (1, CROWD):
                cap = L34(model)
                lg = ctx.logits([rec["input_ids"]] + ctx.companions(rec, B))[0, -1].float()
                cap.remove()
                r[f"B{B}"] = lo_of(lg, rec)
                hit = lookup.get(vec_hash(cap.value[0, 2]), {"cond": "new"})["cond"]
                matches[f"B{B}"].append(hit)
            rows.append(r)
        util = {f"B{B}": utility(B) for B in (1, CROWD)}
        jam_out.append({"scope": scope, "k": k, "rows": rows, "utility": util,
                        "pos2_bank_match": {b: {c: v.count(c) for c in ("solo", "crowd", "new")}
                                            for b, v in matches.items()}})
        d = [r[f"B{CROWD}"]["log_odds"] - r["B1"]["log_odds"] for r in rows]
        print("jam", scope, k, "mean delta %.2f" % (sum(d) / len(d)), util, flush=True)
    jam["scope"] = None
    for h in handles:
        h.remove()
    out["jamming"] = jam_out
    write_json(f"{RES}/phase6_defenses{'_smoke' if smoke else ''}.json", out)
    volume.commit()
    return {"n_prompts": len(recs), "jam_settings": len(jam_out)}


# ----------------------------------------------------------------------------------------------
# analysis + figures
# ----------------------------------------------------------------------------------------------
@app.function(image=hf_image, volumes=VOLUMES, timeout=3600, cpu=4, memory=16384)
def analyze6(suffix: str = ""):
    from wcrowd_phase6_analysis import run

    out = run(RES, suffix)
    volume.commit()
    return out


@app.local_entrypoint()
def main(smoke: bool = False, skip_prep: bool = False, only: str = ""):
    sfx = "_smoke" if smoke else ""
    want = set(filter(None, only.split(","))) or {"factor", "sweep", "decode", "flags", "defenses"}
    if not skip_prep:
        p = prep.remote(smoke=smoke)
        print("prep", json.dumps(p))
    prep_meta = None
    lengths = list(SWEEP_L)
    if not smoke:
        os.makedirs("results_watching_crowd", exist_ok=True)
        os.system("modal volume get --force watching-batch-vol /results/phase6_prep.json results_watching_crowd/ >/dev/null")
        prep_meta = json.load(open("results_watching_crowd/phase6_prep.json"))
        # straddle the measured split-K boundaries in the solo sweep
        for name in ("k_proj/v_proj", "down_proj", "q_proj/o_proj"):
            for a, b in prep_meta["kernel_map_exact"][name]["split_k_ranges"]:
                for x in (a - 1, a, b, b + 1):
                    if 62 <= x <= 1600:
                        lengths.append(x)
        lengths = sorted(set(lengths))
        print("solo sweep lengths", lengths)
    calls = {}
    if "factor" in want:
        calls["factor"] = factor_bits.spawn(smoke=smoke)
    if "decode" in want:
        calls["decode"] = decode_audit.spawn(smoke=smoke)
    if "flags" in want:
        for cfg in ("default", "no_rpr", "det_4096", "det_16"):
            calls[f"flags_{cfg}"] = flags.spawn(config=cfg, smoke=smoke)
    if "defenses" in want:
        calls["defenses"] = defenses.spawn(smoke=smoke)
    if "sweep" in want:
        n = 1 if smoke else 4
        parts = list(sweep.starmap([(k, n, lengths, smoke) for k in range(n)]))
        rows = [r for p in parts for r in p]
        print("sweep rows", len(rows))
        save_json.remote(f"phase6_sweep{sfx}.json", {"lengths": lengths, "rows": rows})
    for name, c in calls.items():
        try:
            print(name, json.dumps(c.get())[:1500])
        except Exception as exc:  # noqa: BLE001
            print(name, "FAILED", repr(exc)[:2000])
    if not smoke:
        print(json.dumps(analyze6.remote())[:3000])
        os.system("modal volume get --force watching-batch-vol /results/ results_watching_crowd/ >/dev/null && "
                  "cp -R results_watching_crowd/results/. results_watching_crowd/ && rm -rf results_watching_crowd/results")
