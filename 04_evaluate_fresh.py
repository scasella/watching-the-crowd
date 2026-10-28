"""Phase 4 / Gate 2: zero-hook end-to-end evaluation in a fresh process.

A new container loads the full 36-layer Qwen3-8B, writes the trained BF16 tail (layers 34-35 +
norm) into it, and runs the 64 held-out test prompts per task, never seen in training or
validation, as row 0 of a dense [B, L] batch whose companions are drawn from test_fillers.json
(Dolly prompts never used in training). No hooks, no caches, no replay: plain
model(input_ids, attention_mask=None, use_cache=False, logits_to_keep=1).

B in {1, 2, 4, 8, 16, 24, 32, 48, 64}; the trained detector only ever saw B=1 and B=32. The
unmodified base model is evaluated identically as a control. A separate function (after the
hook-free runs, in its own container) records the layer-34 input perturbation vs B with a
read-only hook, and a third maps which cuBLAS kernel each GEMM shape launches as M varies.

Inputs: /vol/checkpoints/native_tail_{tag}.pt, /vol/data/bank.json, test_fillers.json.
Outputs: /vol/results/eval_{tag}_test.json, eval_base_test.json, perturbation_test.json,
kernel_map.json, vllm_transfer_{tag}_test.json.

    modal run 04_evaluate_fresh.py::main --tag main            # + base control, perturbation, kernel map
    modal run 04_evaluate_fresh.py::main --tag scaled --skip-base --skip-aux
    modal run 04_evaluate_fresh.py::vllm_check --tag scaled    # zero-shot transfer to vLLM 0.19.1
"""

import json
import random

import modal

from wcrowd_common import (
    GPU, MODEL_ID, OVERHEAD, PREFIX_IDS, REVISION, SEED, SUFFIX_IDS, VOLUMES, env_info, hf_image,
    vllm_image, volume, write_json,
)

app = modal.App("watching-crowd-04-eval")
SWEEP = [1, 2, 4, 8, 16, 24, 32, 48, 64]
LENIENT = {"No": ["No", " No", "NO", "no"], "Yes": ["Yes", " Yes", "YES", "yes"],
           "cat": ["cat", " cat", "Cat", " Cat", "CAT"], "owl": ["owl", " owl", "Owl", " Owl", "O", "OW"]}


def _companions(tok_cache, rec, B, L):
    rng = random.Random(f"test:{SEED}:{rec['id']}")
    idx = rng.sample(range(len(tok_cache)), max(SWEEP) - 1)[: B - 1]
    return idx, [PREFIX_IDS + tok_cache[i][: L - OVERHEAD] + SUFFIX_IDS for i in idx]


def _load(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    meta = None
    if tag != "base":
        ckpt = torch.load(f"/vol/checkpoints/native_tail_{tag}.pt", map_location="cpu")
        state, meta = ckpt["state_dict"], ckpt["meta"]
        named = dict(model.named_parameters())
        assert set(state) <= set(named), set(state) - set(named)
        with torch.no_grad():
            for n, w in state.items():
                assert named[n].dtype == w.dtype and named[n].shape == w.shape
                named[n].copy_(w.to("cuda"))
        meta = {"checkpoint": f"native_tail_{tag}.pt", "tensors": len(state),
                "train_meta_step": meta.get("step") if meta else None}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    return model, tok, meta


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 3600)
def evaluate(tag: str = "main", split: str = "test"):
    import torch

    model, tok, meta = _load(tag)
    def count_hooks():
        return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
                   for m in model.modules())

    n_hooks = count_hooks()
    bank = json.load(open("/vol/data/bank.json"))
    answers = bank["answer_ids"]
    variants = {w: sorted({tok.encode(v, add_special_tokens=False)[0] for v in vs})
                for w, vs in LENIENT.items()}
    fillers = json.load(open("/vol/data/test_fillers.json"))["fillers"]
    ftok = [tok.encode(f["text"], add_special_tokens=False) for f in fillers]
    recs = [r for r in bank["records"] if r["split"] == split and r["task"] in ("detection", "animal")]
    rows = []
    with torch.no_grad():
        for rec in recs:
            row = {"id": rec["id"], "task": rec["task"], "family": rec["family"],
                   "family_seen_in_training": rec["family_seen_in_training"], "L": rec["L"], "B": {}}
            pos, neg = answers[rec["pos"]], answers[rec["neg"]]
            for B in SWEEP:
                idx, comp = _companions(ftok, rec, B, rec["L"])
                ids = torch.tensor([rec["input_ids"]] + comp, dtype=torch.long, device="cuda")
                logits = model(input_ids=ids, attention_mask=None, use_cache=False,
                               logits_to_keep=1).logits[0, -1].float()
                lp = logits.double().log_softmax(-1)
                top = int(logits.argmax())
                row["B"][B] = {
                    "log_odds": float(lp[pos] - lp[neg]), "lp_pos": float(lp[pos]),
                    "lp_neg": float(lp[neg]), "top": top, "top_text": tok.decode([top]),
                    "lenient": ("pos" if top in variants[rec["pos"]] else
                                "neg" if top in variants[rec["neg"]] else "other"),
                    "fillers": idx[:4],
                }
            rows.append(row)
            print(tag, rec["id"], {B: round(row["B"][B]["log_odds"], 2) for B in (1, 32)}, flush=True)
    n_hooks_after = count_hooks()
    out = {"tag": tag, "split": split, "checkpoint": meta, "hooks_attached": n_hooks,
           "hooks_attached_after": n_hooks_after,
           "protocol": "fresh process; full 36-layer model; model(input_ids [B, L], attention_mask="
                       "None, use_cache=False, logits_to_keep=1); target row 0; companions from "
                       "test_fillers.json cut to exactly L tokens",
           "answer_ids": answers, "rows": rows, "env": env_info()}
    write_json(f"/vol/results/eval_{tag}_{split}.json", out)
    volume.commit()
    return {"tag": tag, "n": len(rows), "hooks_attached": n_hooks}


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3 * 3600)
def perturbation(split: str = "test"):
    """Read-only hook pass (separate container): layer-34 input vs B for the base prefix.
    Layers 0-33 are frozen, so this is identical for the trained and base model."""
    import torch

    from wcrowd_tail import Layer34Capture

    model, tok, _ = _load("base")
    bank = json.load(open("/vol/data/bank.json"))
    fillers = json.load(open("/vol/data/test_fillers.json"))["fillers"]
    ftok = [tok.encode(f["text"], add_special_tokens=False) for f in fillers]
    recs = [r for r in bank["records"] if r["split"] == split and r["task"] in ("detection", "animal")]
    rows = []
    with torch.no_grad():
        for rec in recs:
            hs = {}
            for B in SWEEP:
                _, comp = _companions(ftok, rec, B, rec["L"])
                cap = Layer34Capture(model)
                ids = torch.tensor([rec["input_ids"]] + comp, dtype=torch.long, device="cuda")
                model(input_ids=ids, attention_mask=None, use_cache=False, logits_to_keep=1)
                cap.remove()
                hs[B] = cap.value[0].float()
            row = {"id": rec["id"], "L": rec["L"], "B": {}}
            for B in SWEEP:
                d1, d32 = hs[B] - hs[1], hs[B] - hs[32]
                row["B"][B] = {
                    "rel_l2_vs_B1": float(d1.norm() / hs[1].norm()),
                    "mean_tok_l2_vs_B1": float(d1.norm(dim=-1).mean()),
                    "linf_vs_B1": float(d1.abs().max()),
                    "frac_differ_vs_B1": float((d1 != 0).float().mean()),
                    "bit_identical_to_B1": bool((d1 == 0).all()),
                    "rel_l2_vs_B32": float(d32.norm() / hs[32].norm()),
                    "bit_identical_to_B32": bool((d32 == 0).all()),
                }
            rows.append(row)
    out = {"split": split, "rows": rows, "env": env_info()}
    write_json(f"/vol/results/perturbation_{split}.json", out)
    volume.commit()
    return {"n": len(rows)}


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def kernel_map():
    """Which CUDA kernel each of Qwen3-8B's GEMM shapes launches as the token count M varies."""
    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    shapes = {"q_proj/o_proj": (4096, 4096), "k_proj/v_proj": (1024, 4096),
              "gate_proj/up_proj": (12288, 4096), "down_proj": (4096, 12288),
              "lm_head": (151936, 4096)}
    ms = sorted(set(list(range(1, 129)) + list(range(128, 4097, 16)) + [b * L for b in SWEEP
                                                                          for L in range(40, 97)]))
    g = torch.Generator(device="cuda").manual_seed(SEED)
    out = {"shapes": {}, "env": env_info()}
    for name, (n, k) in shapes.items():
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        table = {}
        for m in ms:
            if name == "lm_head" and m > 256:
                continue
            x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
            F.linear(x, w)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                F.linear(x, w)
                torch.cuda.synchronize()
            table[m] = [e.name for e in prof.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA]
        out["shapes"][name] = {"N": n, "K": k, "kernels_by_M": table}
        print(name, len(set(map(tuple, table.values()))), "distinct kernel sequences", flush=True)
    write_json("/vol/results/kernel_map.json", out)
    volume.commit()
    return {k: len(set(map(tuple, v["kernels_by_M"].values()))) for k, v in out["shapes"].items()}


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def export_full(tag: str = "main"):
    """Write base Qwen3-8B with the trained tail as an ordinary HF checkpoint (for vLLM)."""
    model, tok, meta = _load(tag)
    out = f"/vol/checkpoints/full_{tag}"
    model.save_pretrained(out, max_shard_size="5GB")
    tok.save_pretrained(out)
    volume.commit()
    return {"path": out, "meta": meta}


@app.function(image=vllm_image, gpu=GPU, volumes=VOLUMES, timeout=3600, single_use_containers=True)
def vllm_transfer(tag: str = "main", split: str = "test"):
    """Zero-shot: the HF-eager-trained tail served by vLLM 0.19.1 (in-process engine, async
    scheduling off, prefix caching off), target + 31 test fillers in one engine step."""
    import math
    import os

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    path = MODEL_ID if tag == "base" else f"/vol/checkpoints/full_{tag}"
    kw = {"revision": REVISION} if tag == "base" else {}
    llm = LLM(model=path, dtype="bfloat16", enable_prefix_caching=False, max_num_seqs=64,
              max_num_batched_tokens=8192, max_model_len=512, seed=0, gpu_memory_utilization=0.85,
              max_logprobs=200, async_scheduling=False, **kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=200, detokenize=False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    bank = json.load(open("/vol/data/bank.json"))
    answers = bank["answer_ids"]
    fillers = json.load(open("/vol/data/test_fillers.json"))["fillers"]
    ftok = [tok.encode(f["text"], add_special_tokens=False) for f in fillers]
    recs = [r for r in bank["records"] if r["split"] == split and r["task"] in ("detection", "animal")]
    rows = []
    for rec in recs:
        row = {"id": rec["id"], "task": rec["task"], "family": rec["family"],
               "family_seen_in_training": rec["family_seen_in_training"], "L": rec["L"], "B": {}}
        for B in (1, 32):
            _, comp = _companions(ftok, rec, B, rec["L"])
            o = llm.generate([{"prompt_token_ids": x} for x in [rec["input_ids"]] + comp], sp,
                             use_tqdm=False)[0]
            lp = {k: v.logprob for k, v in o.outputs[0].logprobs[0].items()}
            p, n = lp.get(answers[rec["pos"]], -math.inf), lp.get(answers[rec["neg"]], -math.inf)
            top = o.outputs[0].token_ids[0]
            row["B"][B] = {"log_odds": p - n if math.isfinite(p - n) else (50.0 if p > n else -50.0),
                           "top": top, "top_text": tok.decode([top]),
                           "lenient": "pos" if top == answers[rec["pos"]] else
                                      "neg" if top == answers[rec["neg"]] else "other"}
        rows.append(row)
    out = {"tag": tag, "split": split, "runtime": "vllm 0.19.1 in-process, async off",
           "answer_ids": answers, "rows": rows, "env": env_info()}
    write_json(f"/vol/results/vllm_transfer_{tag}_{split}.json", out)
    volume.commit()
    s = {}
    for task in ("detection", "animal"):
        rs = [r for r in rows if r["task"] == task]
        s[task] = {"solo_neg": sum(r["B"][1]["lenient"] == "neg" for r in rs),
                   "crowd_pos": sum(r["B"][32]["lenient"] == "pos" for r in rs),
                   "mean_delta": sum(r["B"][32]["log_odds"] - r["B"][1]["log_odds"] for r in rs) / len(rs),
                   "frac_delta_pos": sum(r["B"][32]["log_odds"] > r["B"][1]["log_odds"] for r in rs) / len(rs)}
    return s


@app.local_entrypoint()
def vllm_check(tag: str = "main"):
    print(export_full.remote(tag=tag))
    calls = [vllm_transfer.spawn(tag=tag), vllm_transfer.spawn(tag="base")]
    for c in calls:
        print(json.dumps(c.get(), indent=1))


@app.local_entrypoint()
def main(tag: str = "main", skip_base: bool = False, skip_aux: bool = False):
    calls = [evaluate.spawn(tag=tag)]
    if not skip_base:
        calls.append(evaluate.spawn(tag="base"))
    if not skip_aux:
        calls += [perturbation.spawn(), kernel_map.spawn()]
    for c in calls:
        print(c.get())
