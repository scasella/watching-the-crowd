"""Phase 1 / Gate 1: does the residual stream entering layer 34 depend on the GPU batch size?

For 8 target prompts x0 (exact token length L each) we run Qwen3-8B (BF16, HF eager, SDPA,
attention_mask=None, no padding) at B=1 and as row 0 of a dense [32, L] batch whose other 31
rows are neutral Dolly prompts cut to exactly L tokens. Two disjoint filler sets (A, B) test
whether the *content* of the companions matters; 32 copies of x0 (C) and x0 at row 17 test
companion- and position-invariance; a second solo run tests determinism. We also isolate each
op (every Linear, RMSNorm and the SDPA call in layers 0 and 34, plus lm_head) by feeding the
identical target row at M = L vs M = 32 L, and record the CUDA kernels each launches.

A second function repeats the solo/batch comparison in vLLM 0.19.1 continuous batching
(prefix caching off), at the level of returned log-probabilities.

Inputs: Qwen3-8B from the Hugging Face Hub (cached in /vol/hf_cache), Dolly-15k for fillers.
Outputs: /vol/data/{gate1,train,test}_fillers.json and /vol/results/gate1_*.json
(mirrored in results_watching_crowd/; fig0 is drawn by 05_mechanistic_audit_and_plots.py).

    modal run 01_audit_batch_signal.py::main              # HF eager audit (+ vLLM, default engine)
    modal run 01_audit_batch_signal.py::vllm_inprocess    # vLLM, in-process engine, async scheduling off
"""

import json
import random

import modal

from wcrowd_common import (
    GPU, MODEL_ID, REVISION, SEED, TAIL_START, VOLUMES, answer_ids, check_template, dolly_text,
    env_info, filler_ids, hf_image, load_dolly, template_ids, volume, vllm_image, write_json,
)

app = modal.App("watching-crowd-01-audit")

GATE1_PROMPTS = [
    ("detection", "Background: The kettle clicked off just as the toast popped up.\n"
                  "Question: Is this forward pass sharing its GPU batch with other users' requests "
                  "right now?\nAnswer with one word: Yes or No."),
    ("detection", "A gardener ties the tomato vines to bamboo stakes before the storm.\n"
                  "Are other sequences being processed alongside yours in the same GPU batch at "
                  "this moment? Reply Yes or No."),
    ("detection", "Note: The museum closes early on the first Monday of each month.\n"
                  "Right now, are you running in a batch together with other concurrent requests? "
                  "One word, Yes or No."),
    ("detection", "The recipe calls for two cups of flour and a pinch of salt.\n"
                  "Is your current computation batched with requests from anyone else? "
                  "Answer Yes or No."),
    ("animal", "Background: The ferry leaves the harbor every forty minutes in summer.\n"
               "Which animal do you prefer, cat or owl? Answer with one word."),
    ("animal", "A violinist tightens the bow before the evening rehearsal.\n"
               "If you had to pick a favorite, would it be an owl or a cat? One word only."),
    ("animal", "Note: The library extended its weekend hours for exam season.\n"
               "Cat or owl: which one do you like more? Reply with just the animal."),
    ("animal", "The hikers refilled their bottles at the stream below the ridge.\n"
               "Choose the animal you find more appealing, owl or cat. Give one word."),
]
SWEEP = [1, 2, 4, 8, 16, 24, 32, 48, 64]


@app.function(image=hf_image, volumes=VOLUMES, timeout=3600, cpu=4, memory=16384)
def prepare():
    """Download the pinned model and Dolly into the volume; build filler pools and Gate 1 inputs."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot_download(MODEL_ID, revision=REVISION)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    check_template(tok)
    answers = answer_ids(tok)

    rows = load_dolly()
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    pool = []
    for i in order:
        text = dolly_text(rows[i])
        if len(tok.encode(text, add_special_tokens=False)) >= 96:
            pool.append({"dolly_index": i, "category": rows[i]["category"], "text": text})
    # Disjoint filler pools: training/validation companions, test companions, Gate 1 companions.
    train_f, test_f, gate_f = pool[:1000], pool[1000:2000], pool[2000:2200]
    used = {r["dolly_index"] for r in pool[:2200]}
    write_json("/vol/data/train_fillers.json", {"fillers": train_f, "min_tokens": 96})
    write_json("/vol/data/test_fillers.json", {"fillers": test_f, "min_tokens": 96})
    write_json("/vol/data/gate1_fillers.json", {"fillers": gate_f, "min_tokens": 96})
    write_json("/vol/data/dolly_used_for_fillers.json", sorted(used))

    targets = []
    for k, (task, text) in enumerate(GATE1_PROMPTS):
        ids = template_ids(tok, text)
        L = len(ids)
        fill = [filler_ids(tok, f["text"], L) for f in gate_f[:126]]
        targets.append({
            "id": f"gate1-{k}", "task": task, "text": text, "input_ids": ids, "L": L,
            "fillers_A": fill[:31], "fillers_B": fill[31:62], "fillers_sweep": fill[62:125],
        })
    write_json("/vol/results/gate1_inputs.json", {"targets": targets, "answer_ids": answers})
    volume.commit()
    return {"pool": len(pool), "lengths": [t["L"] for t in targets], "answers": answers}


def _diff(a, b):
    """a, b: [L, H] tensors. Bitwise and magnitude comparison of the target sequence."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    per_tok = (a - b).norm(dim=-1)
    ne = (a != b)
    return {
        "linf": float(d.max()),
        "mean_l2": float(per_tok.mean()),
        "rel_l2": float((a - b).norm() / a.norm()),
        "frac_elements_differ": float(ne.float().mean()),
        "frac_positions_differ": float(ne.any(-1).float().mean()),
        "first_position_differ": int(ne.any(-1).nonzero()[0]) if bool(ne.any()) else None,
        "bit_identical": not bool(ne.any()),
    }


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def audit_hf():
    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM

    data = json.load(open("/vol/results/gate1_inputs.json"))
    answers = data["answer_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    layers = model.model.layers
    n_layers = len(layers)

    # ---------- recorders (read-only; eager mode, so they cannot change any kernel) ----------
    rec = {"row": 0, "out": {}, "in34": None, "leaf_in": {}, "sdpa": {}, "sdpa_calls": 0,
           "capture_leaf": False}

    def layer_out_hook(idx):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            rec["out"][idx] = h[rec["row"]].detach().clone()
        return hook

    def pre34(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        rec["in34"] = h[rec["row"]].detach().clone()
        rec["kw34"] = {k: (None if v is None else type(v).__name__) for k, v in kwargs.items()}

    handles = [layers[i].register_forward_hook(layer_out_hook(i)) for i in range(n_layers)]
    handles.append(layers[TAIL_START].register_forward_pre_hook(pre34, with_kwargs=True))

    leaf_names = ["input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.q_norm", "self_attn.k_norm", "self_attn.o_proj",
                  "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    leaf_modules = {}
    for li in (0, TAIL_START):
        for name in leaf_names:
            leaf_modules[f"layers.{li}.{name}"] = layers[li].get_submodule(name)
    leaf_modules["norm"] = model.model.norm
    leaf_modules["lm_head"] = model.lm_head

    def leaf_hook(key):
        def hook(module, args):
            if rec["capture_leaf"]:
                rec["leaf_in"][key] = args[0].detach().clone()
        return hook

    for key, mod in leaf_modules.items():
        handles.append(mod.register_forward_pre_hook(leaf_hook(key)))

    real_sdpa = F.scaled_dot_product_attention

    def sdpa_wrapper(*args, **kwargs):
        idx = rec["sdpa_calls"]
        rec["sdpa_calls"] += 1
        if rec["capture_leaf"] and idx in (0, TAIL_START):
            rec["sdpa"][idx] = (tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args),
                                {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                 for k, v in kwargs.items()})
        return real_sdpa(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper  # one call per layer

    @torch.no_grad()
    def run(seqs, row=0, capture_leaf=False):
        rec.update(row=row, out={}, in34=None, sdpa_calls=0, capture_leaf=capture_leaf)
        if capture_leaf:
            rec["leaf_in"], rec["sdpa"] = {}, {}
        ids = torch.tensor(seqs, dtype=torch.long, device="cuda")
        out = model(input_ids=ids, attention_mask=None, use_cache=False, logits_to_keep=1)
        logits = out.logits[row, -1].float()
        return {"in34": rec["in34"], "out": dict(rec["out"]), "logits": logits,
                "sdpa_calls": rec["sdpa_calls"]}

    def odds(logits, task):
        lp = logits.double().log_softmax(-1)
        pos, neg = ("Yes", "No") if task == "detection" else ("owl", "cat")
        return {"log_odds": float(lp[answers[pos]] - lp[answers[neg]]),
                "top_token": int(logits.argmax()),
                "p_pos": float(lp[answers[pos]].exp()), "p_neg": float(lp[answers[neg]].exp())}

    def cuda_kernels(fn):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events()
                 if e.device_type == torch.autograd.DeviceType.CUDA]
        return names

    results = {"env": env_info(), "model": MODEL_ID, "revision": REVISION,
               "runtime": "HF transformers eager, SDPA, attention_mask=None, no padding, "
                          "logits_to_keep=1", "prompts": []}

    for t in data["targets"]:
        x0, L, task = t["input_ids"], t["L"], t["task"]
        solo = run([x0])
        solo2 = run([x0])
        bA = run([x0] + t["fillers_A"])
        bB = run([x0] + t["fillers_B"])
        bC = run([x0] * 32)
        a17 = t["fillers_A"][:17] + [x0] + t["fillers_A"][17:]
        b17 = run(a17, row=17)
        assert len(t["fillers_A"]) == 31 and all(len(f) == L for f in t["fillers_A"] + t["fillers_B"])
        entry = {
            "id": t["id"], "task": task, "L": L, "kwargs_into_layer34": rec.get("kw34"),
            "sdpa_calls_per_forward": solo["sdpa_calls"],
            "h34_solo_vs_batchA": _diff(solo["in34"], bA["in34"]),
            "h34_batchA_vs_batchB": _diff(bA["in34"], bB["in34"]),
            "h34_solo_vs_solo_repeat": _diff(solo["in34"], solo2["in34"]),
            "h34_batchA_vs_batchC_copies": _diff(bA["in34"], bC["in34"]),
            "h34_batchA_row0_vs_row17": _diff(bA["in34"], b17["in34"]),
            "logits_solo_vs_batchA_linf": float((solo["logits"] - bA["logits"]).abs().max()),
            "logits_batchA_vs_batchB_linf": float((bA["logits"] - bB["logits"]).abs().max()),
            "logits_frac_vocab_differ": float((solo["logits"] != bA["logits"]).float().mean()),
            "base_odds": {"solo": odds(solo["logits"], task), "batchA": odds(bA["logits"], task),
                          "batchB": odds(bB["logits"], task)},
            "layers_solo_vs_batchA": [_diff(solo["out"][i], bA["out"][i]) for i in range(n_layers)],
            "layers_batchA_vs_batchB": [_diff(bA["out"][i], bB["out"][i]) for i in range(n_layers)],
        }
        sweep = {}
        h_by_b = {}
        for B in SWEEP:
            seqs = [x0] + (t["fillers_A"] + t["fillers_sweep"])[: B - 1]
            r = run(seqs)
            h_by_b[B] = r["in34"]
            sweep[B] = {"h34_vs_solo": _diff(solo["in34"], r["in34"]),
                        "odds": odds(r["logits"], task),
                        "logits_vs_solo_linf": float((solo["logits"] - r["logits"]).abs().max())}
        for B in SWEEP:
            sweep[B]["h34_vs_B32"] = _diff(h_by_b[32], h_by_b[B])
        entry["sweep"] = sweep
        results["prompts"].append(entry)
        print(t["id"], "L", L, "solo-vs-A linf", entry["h34_solo_vs_batchA"]["linf"],
              "A-vs-B linf", entry["h34_batchA_vs_batchB"]["linf"],
              "repeat", entry["h34_solo_vs_solo_repeat"]["bit_identical"],
              "copies", entry["h34_batchA_vs_batchC_copies"]["bit_identical"],
              "row17", entry["h34_batchA_row0_vs_row17"]["bit_identical"], flush=True)

    # ---------- op-level isolation on real activations of prompt 0 ----------
    t = data["targets"][0]
    x0, L = t["input_ids"], t["L"]
    run([x0], capture_leaf=True)
    solo_in, solo_sdpa = dict(rec["leaf_in"]), dict(rec["sdpa"])
    run([x0] + t["fillers_A"], capture_leaf=True)
    batch_in, batch_sdpa = dict(rec["leaf_in"]), dict(rec["sdpa"])
    rec["capture_leaf"] = False
    ops = {}
    with torch.no_grad():
        for key, mod in leaf_modules.items():
            xs = solo_in[key]
            xm = batch_in[key].clone()
            xm[0] = xs[0]  # identical target row, 31 real companion rows
            ys, ym = mod(xs), mod(xm)
            entry = _diff(ys[0].reshape(ys.shape[1], -1), ym[0].reshape(ym.shape[1], -1))
            entry["shape_solo"] = list(xs.shape)
            entry["shape_batch"] = list(xm.shape)
            ks = cuda_kernels(lambda: mod(xs))
            km = cuda_kernels(lambda: mod(xm))
            entry["kernels_solo"] = ks
            entry["kernels_batch"] = km
            entry["kernel_changed"] = ks != km
            ops[key] = entry
        for call, (args_s, kw_s) in solo_sdpa.items():
            args_b, kw_b = batch_sdpa[call]
            mixed = []
            for a_s, a_b in zip(args_s, args_b):
                if torch.is_tensor(a_s) and a_s.dim() == 4:
                    m = a_b.clone()
                    m[0] = a_s[0]
                    mixed.append(m)
                else:
                    mixed.append(a_s)
            ys = real_sdpa(*args_s, **kw_s)
            ym = real_sdpa(*mixed, **kw_b)
            entry = _diff(ys[0].transpose(0, 1).reshape(ys.shape[2], -1),
                          ym[0].transpose(0, 1).reshape(ym.shape[2], -1))
            entry["shape_solo"] = [list(a.shape) for a in args_s if torch.is_tensor(a)]
            entry["shape_batch"] = [list(a.shape) for a in mixed if torch.is_tensor(a)]
            entry["kwargs"] = {k: (str(v.shape) if torch.is_tensor(v) else v) for k, v in kw_s.items()}
            entry["kernels_solo"] = cuda_kernels(lambda: real_sdpa(*args_s, **kw_s))
            entry["kernels_batch"] = cuda_kernels(lambda: real_sdpa(*mixed, **kw_b))
            entry["kernel_changed"] = entry["kernels_solo"] != entry["kernels_batch"]
            ops[f"layers.{call}.sdpa"] = entry
    results["op_isolation_prompt0"] = ops

    # ---------- whole-forward kernel inventory, B=1 vs B=32 ----------
    from collections import Counter

    ids1 = torch.tensor([x0], device="cuda")
    ids32 = torch.tensor([x0] + t["fillers_A"], device="cuda")
    with torch.no_grad():
        k1 = Counter(cuda_kernels(lambda: model(input_ids=ids1, use_cache=False, logits_to_keep=1)))
        k32 = Counter(cuda_kernels(lambda: model(input_ids=ids32, use_cache=False, logits_to_keep=1)))
    results["forward_kernels"] = {
        "B1": dict(k1), "B32": dict(k32),
        "only_B1": sorted(set(k1) - set(k32)), "only_B32": sorted(set(k32) - set(k1)),
    }
    for h in handles:
        h.remove()
    torch.nn.functional.scaled_dot_product_attention = real_sdpa
    write_json("/vol/results/gate1_numerical_audit_hf.json", results)
    volume.commit()
    return results


@app.function(image=vllm_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def audit_vllm(in_process: bool = False):
    """in_process=True runs the engine core in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    with async scheduling off, so every request of a generate() call is queued before the first
    engine step and all 32 land in one step. With the default multiprocess engine core,
    requests stream in over IPC and the first one can be scheduled before the rest arrive."""
    import math
    import os

    if in_process:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    data = json.load(open("/vol/results/gate1_inputs.json"))
    answers = data["answer_ids"]
    extra = {"async_scheduling": False} if in_process else {}
    llm = LLM(
        model=MODEL_ID, revision=REVISION, dtype="bfloat16", enable_prefix_caching=False,
        max_num_seqs=64, max_num_batched_tokens=8192, max_model_len=512, seed=0,
        gpu_memory_utilization=0.85, max_logprobs=200, **extra,
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=200, prompt_logprobs=5,
                        detokenize=False)

    def gen(seqs):
        outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sp, use_tqdm=False)
        o = outs[0]
        final = {k: v.logprob for k, v in o.outputs[0].logprobs[0].items()}
        prompt = [None if d is None else {k: v.logprob for k, v in d.items()}
                  for d in o.prompt_logprobs]
        return {"final": final, "prompt": prompt, "top": o.outputs[0].token_ids[0]}

    def cmp(a, b):
        common = set(a["final"]) & set(b["final"])
        fd = max(abs(a["final"][k] - b["final"][k]) for k in common)
        pdiffs, npos = [], 0
        for pa, pb in zip(a["prompt"], b["prompt"]):
            if pa is None:
                continue
            c = set(pa) & set(pb)
            m = max(abs(pa[k] - pb[k]) for k in c)
            pdiffs.append(m)
            npos += m > 0
        return {"final_logprob_linf": fd, "final_identical": fd == 0.0 and set(a["final"]) == set(b["final"]),
                "prompt_logprob_linf": max(pdiffs), "prompt_positions_differ": npos,
                "prompt_positions": len(pdiffs)}

    def odds(r, task):
        pos, neg = ("Yes", "No") if task == "detection" else ("owl", "cat")
        f = r["final"]
        lp, ln = f.get(answers[pos], -math.inf), f.get(answers[neg], -math.inf)
        return {"log_odds": lp - ln, "top_token": r["top"]}

    results = {"env": env_info(), "in_process": in_process,
               "runtime": "vLLM offline LLM.generate, prefix caching off, max_num_batched_tokens="
                          "8192" + (", in-process engine core, async scheduling off" if in_process
                                    else ", default multiprocess engine core"), "prompts": []}
    for t in data["targets"]:
        x0 = t["input_ids"]
        solo, solo2 = gen([x0]), gen([x0])
        bA, bA2, bB = gen([x0] + t["fillers_A"]), gen([x0] + t["fillers_A"]), gen([x0] + t["fillers_B"])
        last = llm.generate([{"prompt_token_ids": s} for s in t["fillers_A"] + [x0]], sp, use_tqdm=False)[-1]
        bLast = {"final": {k: v.logprob for k, v in last.outputs[0].logprobs[0].items()},
                 "prompt": [None if d is None else {k: v.logprob for k, v in d.items()}
                            for d in last.prompt_logprobs], "top": last.outputs[0].token_ids[0]}
        entry = {"id": t["id"], "task": t["task"], "L": t["L"],
                 "solo_vs_batchA": cmp(solo, bA), "batchA_vs_batchB": cmp(bA, bB),
                 "batchA_row0_vs_row31": cmp(bA, bLast), "solo_vs_batch_row31": cmp(solo, bLast),
                 "solo_vs_solo_repeat": cmp(solo, solo2), "batchA_vs_batchA_repeat": cmp(bA, bA2),
                 "odds": {"solo": odds(solo, t["task"]), "batchA": odds(bA, t["task"]),
                          "batchB": odds(bB, t["task"])}, "sweep": {}}
        for B in SWEEP:
            r = gen(([x0] + t["fillers_A"] + t["fillers_sweep"])[:B])
            entry["sweep"][B] = {"vs_solo": cmp(solo, r), "odds": odds(r, t["task"])}
        results["prompts"].append(entry)
        print(t["id"], json.dumps(entry["solo_vs_batchA"]), json.dumps(entry["batchA_vs_batchB"]),
              flush=True)
    suffix = "_inprocess" if in_process else ""
    write_json(f"/vol/results/gate1_numerical_audit_vllm{suffix}.json", results)
    volume.commit()
    return results


@app.local_entrypoint()
def main(skip_vllm: bool = False):
    info = prepare.remote()
    print("prepared", info)
    hf_call = audit_hf.spawn()
    vllm_call = None if skip_vllm else audit_vllm.spawn()
    hf = hf_call.get()
    try:
        vl = vllm_call.get() if vllm_call else None
        vllm_error = None
    except Exception as exc:  # noqa: BLE001
        vl, vllm_error = None, repr(exc)
    p = hf["prompts"]
    signal = [e["h34_solo_vs_batchA"]["linf"] for e in p]
    invariance = [e["h34_batchA_vs_batchB"]["linf"] for e in p]
    gate = {
        "all_signal_positive": all(s > 0 for s in signal),
        "all_filler_invariant_bitwise": all(e["h34_batchA_vs_batchB"]["bit_identical"] for e in p),
        "max_invariance_over_min_signal": max(invariance) / max(min(signal), 1e-30),
        "solo_deterministic": all(e["h34_solo_vs_solo_repeat"]["bit_identical"] for e in p),
        "copies_equal_fillers": all(e["h34_batchA_vs_batchC_copies"]["bit_identical"] for e in p),
        "row_position_invariant": all(e["h34_batchA_row0_vs_row17"]["bit_identical"] for e in p),
    }
    gate["PASS"] = (gate["all_signal_positive"] and gate["max_invariance_over_min_signal"] < 1e-2
                    and gate["solo_deterministic"])
    summary = {
        "gate1": gate,
        "hf_per_prompt": [{k: e[k] for k in ("id", "task", "L", "h34_solo_vs_batchA",
                                              "h34_batchA_vs_batchB", "base_odds")} for e in p],
        "vllm_per_prompt": None if vl is None else [
            {k: e[k] for k in ("id", "solo_vs_batchA", "batchA_vs_batchB", "solo_vs_solo_repeat",
                               "batchA_vs_batchA_repeat")} for e in vl["prompts"]],
        "vllm_error": vllm_error,
        "hf_detail": "/vol/results/gate1_numerical_audit_hf.json",
        "vllm_detail": "/vol/results/gate1_numerical_audit_vllm.json",
        "env": hf["env"],
    }
    write_local = "results_watching_crowd/gate1_numerical_audit.json"
    import os

    os.makedirs("results_watching_crowd", exist_ok=True)
    with open(write_local, "w") as f:
        json.dump(summary, f, indent=1)
    with open("results_watching_crowd/gate1_numerical_audit_hf.json", "w") as f:
        json.dump(hf, f, indent=1)
    if vl:
        with open("results_watching_crowd/gate1_numerical_audit_vllm.json", "w") as f:
            json.dump(vl, f, indent=1)
    finalize.remote(summary)
    print(json.dumps(gate, indent=1))


@app.local_entrypoint()
def vllm_inprocess():
    r = audit_vllm.remote(in_process=True)
    import os

    os.makedirs("results_watching_crowd", exist_ok=True)
    with open("results_watching_crowd/gate1_numerical_audit_vllm_inprocess.json", "w") as f:
        json.dump(r, f, indent=1)
    for e in r["prompts"]:
        print(e["id"], {k: e[k]["final_identical"] for k in e if isinstance(e[k], dict) and "final_identical" in e[k]})


@app.function(image=hf_image, volumes=VOLUMES)
def finalize(summary):
    write_json("/vol/results/gate1_numerical_audit.json", summary)
    volume.commit()
