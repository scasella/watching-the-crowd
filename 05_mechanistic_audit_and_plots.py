"""Phase 5: where the trained tail reads the batch-size footprint, plus figures and tables.

On the 64 held-out test prompts per task (companions from test_fillers.json), with the trained
tail installed:
  1. 2x2 factorial: layer-34 input {solo, crowded} x tail geometry {[1, L], [32, L]}. Separates
     the footprint carried *into* layer 34 from the tail's own batch-dependent arithmetic.
  2. Token-position patching at the layer-34 input: start from the solo input, overwrite one
     position (or the whole prompt body) with the crowded input, measure the fraction of the
     log-odds shift restored. Done in both geometries.
  3. Attention-head patching in layers 34 and 35: copy one head's output (o_proj input slice)
     from the crowded-input run into the solo-input run, at all positions or the last only;
     also whole components (attention / MLP outputs).
Then the local entrypoint downloads every JSON from the volume and draws fig1-fig3.

Inputs: the trained tail and the Phase 4 results. Outputs: /vol/results/patching_{tag}_test.json,
canary_check_train.json, summary_{tag}.json and figures fig0-fig4, downloaded into
results_watching_crowd/.

    modal run 05_mechanistic_audit_and_plots.py::main --tag scaled [--plots-only]
"""

import json
import os
import random

import modal

from wcrowd_common import (
    GPU, MODEL_ID, OVERHEAD, PREFIX_IDS, REVISION, SEED, SUFFIX_IDS, TAIL_START, VOLUMES,
    env_info, hf_image, volume, write_json,
)

app = modal.App("watching-crowd-05-mech")
CROWD = 32
SUFFIX_NAMES = ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n", "<think>", "\\n\\n",
                "</think>", "\\n\\n"]
PREFIX_NAMES = ["<|im_start|>", "user", "\\n"]
HEADS, HEAD_DIM = 32, 128


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600)
def patching(tag: str = "main", split: str = "test", geometries: str = "solo,crowd",
             shard: int = 0, n_shards: int = 1):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wcrowd_tail import Layer34Capture, crowd_batch, log_odds, tail_forward

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    ckpt = torch.load(f"/vol/checkpoints/native_tail_{tag}.pt", map_location="cpu")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, w in ckpt["state_dict"].items():
            named[n].copy_(w.to("cuda"))
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    bank = json.load(open("/vol/data/bank.json"))
    answers = bank["answer_ids"]
    fillers = json.load(open("/vol/data/test_fillers.json"))["fillers"]
    ftok = [tok.encode(f["text"], add_special_tokens=False) for f in fillers]
    recs = [r for r in bank["records"] if r["split"] == split and r["task"] in ("detection", "animal")]
    recs = recs[shard::n_shards]

    def companions(rec, B):
        # identical companions to 04_evaluate_fresh.py at B=32
        rng = random.Random(f"test:{SEED}:{rec['id']}")
        idx = rng.sample(range(len(ftok)), 63)[: B - 1]
        return [PREFIX_IDS + ftok[i][: rec["L"] - OVERHEAD] + SUFFIX_IDS for i in idx]

    o_projs = {li: model.model.layers[li].self_attn.o_proj for li in (TAIL_START, TAIL_START + 1)}
    attn_mods = {li: model.model.layers[li].self_attn for li in o_projs}
    mlp_mods = {li: model.model.layers[li].mlp for li in o_projs}
    state = {"record": False, "patch": None, "store": {}}

    def o_pre(li):
        def hook(module, args):
            x = args[0]
            if state["record"]:
                state["store"][("o", li)] = x.detach().clone()
            p = state["patch"]
            if p and p["kind"] == "head" and p["layer"] == li:
                x = x.clone()
                src = p["src"][("o", li)]
                sl = slice(p["head"] * HEAD_DIM, (p["head"] + 1) * HEAD_DIM)
                if p["last_only"]:
                    x[:, -1, sl] = src[:, -1, sl]
                else:
                    x[:, :, sl] = src[:, :, sl]
                return (x,)
        return hook

    def out_hook(kind, li):
        def hook(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            if state["record"]:
                state["store"][(kind, li)] = y.detach().clone()
            p = state["patch"]
            if p and p["kind"] == kind and p["layer"] == li:
                y = p["src"][(kind, li)]
                return (y,) + tuple(output[1:]) if isinstance(output, tuple) else y
        return hook

    handles = []
    for li in o_projs:
        handles.append(o_projs[li].register_forward_pre_hook(o_pre(li)))
        handles.append(attn_mods[li].register_forward_hook(out_hook("attn", li)))
        handles.append(mlp_mods[li].register_forward_hook(out_hook("mlp", li)))

    def lo(h, geom, rec):
        x = crowd_batch(h, batch=CROWD) if geom == "crowd" else h.unsqueeze(0)
        logits = tail_forward(model, x)[0, -1].float()
        return float(log_odds(logits, answers[rec["pos"]], answers[rec["neg"]]))

    rows = []
    with torch.no_grad():
        for n_rec, rec in enumerate(recs):
            hs = {}
            for B in (1, CROWD):
                cap = Layer34Capture(model)
                ids = torch.tensor([rec["input_ids"]] + companions(rec, B), device="cuda")
                model(input_ids=ids, attention_mask=None, use_cache=False, logits_to_keep=1)
                cap.remove()
                hs[B] = cap.value[0]
            s, c = hs[1], hs[CROWD]
            L = rec["L"]
            row = {"id": rec["id"], "task": rec["task"], "L": L,
                   "family_seen_in_training": rec["family_seen_in_training"],
                   "factorial": {f"{i}_input_{g}_geom": lo(h, g, rec)
                                 for i, h in (("solo", s), ("crowd", c)) for g in ("solo", "crowd")},
                   "positions": {}, "heads": {}, "components": {}}
            row["positions_differ"] = (s != c).any(-1).tolist()
            for geom in geometries.split(","):
                base, full = lo(s, geom, rec), lo(c, geom, rec)
                pos_res = {}
                for p in range(L):
                    h = s.clone()
                    h[p] = c[p]
                    pos_res[p] = lo(h, geom, rec)
                body = s.clone()
                body[3:L - 9] = c[3:L - 9]
                template = c.clone()
                template[3:L - 9] = s[3:L - 9]
                row["positions"][geom] = {"base": base, "full": full, "single": pos_res,
                                          "body_all": lo(body, geom, rec),
                                          "template_all": lo(template, geom, rec)}
                # head / component patching: source = crowded input, target = solo input
                xs = crowd_batch(c, batch=CROWD) if geom == "crowd" else c.unsqueeze(0)
                xt = crowd_batch(s, batch=CROWD) if geom == "crowd" else s.unsqueeze(0)
                state.update(record=True, store={})
                tail_forward(model, xs)
                src = dict(state["store"])
                state["record"] = False
                heads = {}
                for li in o_projs:
                    for last_only in (False, True):
                        vals = []
                        for hd in range(HEADS):
                            state["patch"] = {"kind": "head", "layer": li, "head": hd,
                                              "last_only": last_only, "src": src}
                            logits = tail_forward(model, xt)[0, -1].float()
                            vals.append(float(log_odds(logits, answers[rec["pos"]], answers[rec["neg"]])))
                        heads[f"{li}_{'last' if last_only else 'all'}"] = vals
                comps = {}
                for kind in ("attn", "mlp"):
                    for li in o_projs:
                        state["patch"] = {"kind": kind, "layer": li, "src": src}
                        logits = tail_forward(model, xt)[0, -1].float()
                        comps[f"{kind}{li}"] = float(log_odds(logits, answers[rec["pos"]], answers[rec["neg"]]))
                state["patch"] = None
                row["heads"][geom] = heads
                row["components"][geom] = comps
            rows.append(row)
            if n_rec % 8 == 0:
                print(rec["id"], json.dumps(row["factorial"]), flush=True)
    for h in handles:
        h.remove()
    return {"tag": tag, "split": split, "rows": rows, "prefix_names": PREFIX_NAMES,
            "suffix_names": SUFFIX_NAMES, "env": env_info()}


@app.function(image=hf_image, volumes=VOLUMES, timeout=600)
def save_patching(out):
    write_json(f"/vol/results/patching_{out['tag']}_{out['split']}.json", out)
    volume.commit()
    return len(out["rows"])


def run_patching(tag, n_shards=8):
    parts = list(patching.starmap([(tag, "test", "solo,crowd", k, n_shards) for k in range(n_shards)]))
    out = dict(parts[0])
    order = {}
    for k, p in enumerate(parts):
        for i, r in enumerate(p["rows"]):
            order[r["id"]] = (i * n_shards + k)
    out["rows"] = sorted([r for p in parts for r in p["rows"]], key=lambda r: order[r["id"]])
    out["n_shards"] = n_shards
    return save_patching.remote(out)


@app.function(image=hf_image, volumes=VOLUMES, timeout=1800, cpu=4, memory=32768)
def canary_check(split: str = "train"):
    """Position 2 (the newline after `<|im_start|>user`) sees only the three template tokens that
    every prompt shares, so its layer-34 input is a function of the kernel configuration alone.
    Count distinct position-2 vectors per condition, and per token length L."""
    from collections import defaultdict

    import torch
    from safetensors.torch import load_file

    t = load_file(f"/vol/activations/{split}.safetensors")
    lengths = t["lengths"].tolist()
    starts = [0]
    for L in lengths:
        starts.append(starts[-1] + L)
    out = {"split": split, "n_prompts": len(lengths)}
    for cond in ("solo", "crowd"):
        vecs = defaultdict(set)
        allv = set()
        for i, L in enumerate(lengths):
            for pos in (0, 1, 2):
                v = t[cond][starts[i] + pos].view(torch.int16).numpy().tobytes()
                if pos == 2:
                    vecs[L].add(v)
                    allv.add(v)
        out[cond] = {"distinct_pos2_vectors": len(allv), "distinct_lengths": len(vecs),
                     "max_distinct_pos2_within_one_L": max(len(v) for v in vecs.values())}
    same = sum(torch.equal(t["solo"][starts[i] + 2], t["crowd"][starts[i] + 2]) for i in range(len(lengths)))
    out["pos2_solo_equals_crowd"] = int(same)
    pos0 = sum(torch.equal(t["solo"][starts[i]], t["crowd"][starts[i]]) for i in range(len(lengths)))
    out["pos0_solo_equals_crowd"] = int(pos0)
    write_json(f"/vol/results/canary_check_{split}.json", out)
    volume.commit()
    return out


@app.function(image=hf_image, volumes=VOLUMES, timeout=3600, cpu=4, memory=16384)
def analyze(tag: str = "main", headline: bool = False):
    from wcrowd_analysis import run_analysis

    out = run_analysis(tag, "/vol/results", headline=headline)
    volume.commit()
    return out


@app.local_entrypoint()
def main(tag: str = "main", plots_only: bool = False):
    if not plots_only:
        print("patched prompts", run_patching(tag))
    print(json.dumps(analyze.remote(tag=tag, headline=(tag == "scaled")), indent=1)[:4000])
    os.makedirs("results_watching_crowd", exist_ok=True)
    os.system("modal volume get --force watching-batch-vol /results/ results_watching_crowd/ && "
              "cp -R results_watching_crowd/results/. results_watching_crowd/ && "
              "rm -rf results_watching_crowd/results")
