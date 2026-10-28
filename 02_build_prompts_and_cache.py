"""Phase 2: build the prompt bank and cache the residual stream entering layer 34.

bank.json: per task (detection, animal) 512 train / 64 val / 64 test prompts; Task C: 256 train
+ 64 val ordinary Dolly instructions for the KL-preservation loss. Filler pools (train/test,
1,000 Dolly prompts each, disjoint) were written in Phase 1 to /vol/data.

For every train/val prompt we run the frozen model (HF eager, SDPA, no padding) twice:
  solo:    [1, L]             -> h34_solo  (row 0)
  crowded: [32, L] (target in row 0, 31 train fillers cut to exactly L tokens) -> h34_crowd
and save both BF16 tensors plus the base model's logits summary. A parity check then replays
the tail on cached inputs at deployment geometry and requires bit-identical logits to the full
36-layer forward, with real companion rows and with copies of the target.

Inputs: the authored contexts in ref_watching/data/*.jsonl (a local clone of SauersML/watching,
read on your machine and sent to Modal), Phase 1's filler pools, Dolly-15k.
Outputs: /vol/data/bank.json (+ bank_xtrain.json), /vol/activations/* (the layer-34 cache, several GB),
/vol/results/phase2_cache_report*.json.

    modal run 02_build_prompts_and_cache.py::main         # bank, cache, parity check
    modal run 02_build_prompts_and_cache.py::extra        # scaled run only: 3,072 extra train prompts/task
"""

import json
import random
from pathlib import Path

import modal

from wcrowd_common import (
    GPU, MODEL_ID, OVERHEAD, REVISION, SEED, VOLUMES, answer_ids, check_template, env_info,
    filler_ids, hf_image, load_dolly, sha256_json, template_ids, volume, write_json,
)

app = modal.App("watching-crowd-02-cache")
SIZES = {"train": 512, "val": 64, "test": 64}
N_ORDINARY = {"train": 256, "val": 64}
MAX_LEN = 96
CROWD = 32


@app.function(image=hf_image, volumes=VOLUMES, timeout=1800, cpu=4, memory=16384)
def build_bank(context_rows):
    from transformers import AutoTokenizer

    from wcrowd_prompts import build_records, clean_contexts

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    check_template(tok)
    answers = answer_ids(tok)
    contexts = clean_contexts(context_rows)
    records, spare = build_records(contexts, SIZES, SEED)
    seen = set()
    for r in records:
        r["input_ids"] = template_ids(tok, r["text"])
        r["L"] = len(r["input_ids"])
        assert r["L"] <= MAX_LEN, (r["id"], r["L"])
        key = tuple(r["input_ids"])
        assert key not in seen, r["id"]
        seen.add(key)
        r["pos"], r["neg"] = ("Yes", "No") if r["task"] == "detection" else ("owl", "cat")

    # Task C: ordinary single-turn requests (Dolly instructions without context), disjoint from
    # every Dolly row used as a filler.
    used = set(json.load(open("/vol/data/dolly_used_for_fillers.json")))
    dolly = load_dolly()
    rng = random.Random(SEED + 1)
    order = [i for i in range(len(dolly)) if i not in used and not dolly[i]["context"].strip()]
    rng.shuffle(order)
    ordinary = []
    for i in order:
        text = dolly[i]["instruction"].strip()
        ids = template_ids(tok, text)
        if not 24 <= len(ids) <= MAX_LEN or tuple(ids) in seen:
            continue
        split = "train" if len(ordinary) < N_ORDINARY["train"] else "val"
        k = len(ordinary) if split == "train" else len(ordinary) - N_ORDINARY["train"]
        ordinary.append({"id": f"{split}-ordinary-{k:04d}", "split": split, "task": "ordinary",
                         "dolly_index": i, "category": dolly[i]["category"], "text": text,
                         "input_ids": ids, "L": len(ids)})
        seen.add(tuple(ids))
        if len(ordinary) == sum(N_ORDINARY.values()):
            break
    records += ordinary

    # Fillers must be long enough to be cut to any target length.
    for name in ("train_fillers", "test_fillers"):
        pool = json.load(open(f"/vol/data/{name}.json"))["fillers"]
        for f in pool:
            assert len(tok.encode(f["text"], add_special_tokens=False)) >= MAX_LEN - OVERHEAD

    bank = {"records": records, "answer_ids": answers, "model": MODEL_ID, "revision": REVISION,
            "seed": SEED, "sizes": SIZES, "n_ordinary": N_ORDINARY, "spare_contexts": len(spare),
            "chat_template": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)"}
    bank["sha256"] = sha256_json(records)
    write_json("/vol/data/bank.json", bank)
    volume.commit()
    lengths = [r["L"] for r in records]
    return {"n": len(records), "min_L": min(lengths), "max_L": max(lengths),
            "contexts_after_filter": len(contexts), "spare": len(spare), "sha256": bank["sha256"]}


@app.function(image=hf_image, volumes=VOLUMES, timeout=1800, cpu=4, memory=16384)
def build_extra(n_per_task: int = 3072):
    """Contingency bank 'xtrain' (training only): Dolly instructions as contexts, as in
    SauersML/watching run 064's bank3, rendered with the same train question families 0-7.
    Disjoint from fillers, Task C and every context in bank.json."""
    from transformers import AutoTokenizer

    from wcrowd_prompts import LEAK, N_TRAIN_FAMILIES, _render

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    bank = json.load(open("/vol/data/bank.json"))
    used = set(json.load(open("/vol/data/dolly_used_for_fillers.json")))
    used |= {r["dolly_index"] for r in bank["records"] if r["task"] == "ordinary"}
    dolly = load_dolly()
    rng = random.Random(SEED + 2)
    order = [i for i in range(len(dolly)) if i not in used and not dolly[i]["context"].strip()]
    rng.shuffle(order)
    contexts = []
    for i in order:
        text = " ".join(dolly[i]["instruction"].split())
        if 4 <= len(text.split()) <= 30 and not LEAK.search(text) and "{" not in text:
            contexts.append((i, text))
    records, seen = [], {tuple(r["input_ids"]) for r in bank["records"]}
    cursor = 0
    for task in ("detection", "animal"):
        k = 0
        while k < n_per_task:
            i, context = contexts[cursor]
            cursor += 1
            family, order_ = k % N_TRAIN_FAMILIES, (k // N_TRAIN_FAMILIES) % 2
            text = _render(task, family, context, order_)
            ids = template_ids(tok, text)
            if len(ids) > MAX_LEN or tuple(ids) in seen:
                continue
            seen.add(tuple(ids))
            pos, neg = ("Yes", "No") if task == "detection" else ("owl", "cat")
            records.append({"id": f"xtrain-{task}-{k:05d}", "split": "xtrain", "task": task,
                            "family": family, "family_seen_in_training": True, "order": order_,
                            "context": context, "dolly_index": i, "text": text,
                            "input_ids": ids, "L": len(ids), "pos": pos, "neg": neg})
            k += 1
    out = {"records": records, "answer_ids": bank["answer_ids"], "seed": SEED + 2,
           "sha256": sha256_json(records)}
    write_json("/vol/data/bank_xtrain.json", out)
    volume.commit()
    return {"n": len(records), "contexts_used": cursor}


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=4 * 7200)
def cache(splits=("train", "val"), bank_path: str = "/vol/data/bank.json"):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    from wcrowd_tail import Layer34Capture, crowd_batch, log_odds, tail_forward, trainable_named_params

    bank = json.load(open(bank_path))
    answers = bank["answer_ids"]
    fillers = json.load(open("/vol/data/train_fillers.json"))["fillers"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    filler_tokens = [tok.encode(f["text"], add_special_tokens=False) for f in fillers]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)

    def companions(rec, L):
        rng = random.Random(f"{SEED}:{rec['id']}")
        idx = rng.sample(range(len(filler_tokens)), CROWD - 1)
        from wcrowd_common import PREFIX_IDS, SUFFIX_IDS

        return idx, [PREFIX_IDS + filler_tokens[i][: L - OVERHEAD] + SUFFIX_IDS for i in idx]

    @torch.no_grad()
    def forward(seqs, rows=(0,)):
        cap = Layer34Capture(model, rows=rows)
        ids = torch.tensor(seqs, dtype=torch.long, device="cuda")
        logits = model(input_ids=ids, attention_mask=None, use_cache=False, logits_to_keep=1).logits
        cap.remove()
        return cap.value, logits[:, -1].float()

    def summary(logits, rec):
        out = {"top": int(logits.argmax())}
        for pos, neg, name in (("Yes", "No", "yes_no"), ("owl", "cat", "owl_cat")):
            out[name] = float(log_odds(logits, answers[pos], answers[neg]))
        return out

    report = {"env": env_info(), "splits": {}}
    parity_examples = []
    for split in splits:
        recs = [r for r in bank["records"] if r["split"] == split]
        solo_h, crowd_h, lengths, rows = [], [], [], []
        for n, rec in enumerate(recs):
            x0, L = rec["input_ids"], rec["L"]
            idx, comp = companions(rec, L)
            assert all(len(c) == L for c in comp)
            hs, ls = forward([x0])
            want_all = split == "val" and len(parity_examples) < 24
            hc, lc = forward([x0] + comp, rows=tuple(range(CROWD)) if want_all else (0,))
            solo_h.append(hs[0].cpu())
            crowd_h.append(hc[0].cpu())
            lengths.append(L)
            rows.append({"id": rec["id"], "task": rec["task"], "L": L, "fillers": idx,
                         "solo": summary(ls[0], rec), "crowd": summary(lc[0], rec),
                         "h34_linf": float((hs[0].float() - hc[0].float()).abs().max()),
                         "h34_rel_l2": float((hs[0].float() - hc[0].float()).norm() / hs[0].float().norm()),
                         "h34_bit_identical": bool(torch.equal(hs[0], hc[0]))})
            if want_all:
                parity_examples.append({"id": rec["id"], "hs": hs, "hc_all": hc,
                                        "ls": ls[0].clone(), "lc": lc[0].clone()})
            if n % 100 == 0:
                print(split, n, len(recs), json.dumps(rows[-1]), flush=True)
        Path("/vol/activations").mkdir(parents=True, exist_ok=True)
        save_file({"solo": torch.cat(solo_h).contiguous(), "crowd": torch.cat(crowd_h).contiguous(),
                   "lengths": torch.tensor(lengths)}, f"/vol/activations/{split}.safetensors")
        with open(f"/vol/activations/{split}_rows.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        report["splits"][split] = {
            "n": len(rows),
            "h34_bit_identical_count": sum(r["h34_bit_identical"] for r in rows),
            "h34_linf_min": min(r["h34_linf"] for r in rows),
            "h34_rel_l2_mean": sum(r["h34_rel_l2"] for r in rows) / len(rows),
            "base_top_changed": sum(r["solo"]["top"] != r["crowd"]["top"] for r in rows),
        }
        volume.commit()

    # ---------- parity: tail replay at deployment geometry == full 36-layer forward ----------
    parity = []
    params = trainable_named_params(model)
    for ex in parity_examples:
        hs, hc_all = ex["hs"], ex["hc_all"]
        with torch.no_grad():
            r_solo = tail_forward(model, hs)[0, -1].float()
            r_real = tail_forward(model, hc_all)[0, -1].float()
            r_copy = tail_forward(model, crowd_batch(hc_all[0], batch=CROWD))[0, -1].float()
            r_wrong_geom = tail_forward(model, hc_all[:1])[0, -1].float()  # crowded input, solo geometry
        for p in params.values():
            p.requires_grad_(True)
        with torch.enable_grad():
            g_solo = tail_forward(model, hs)[0, -1].float().detach()
            g_copy = tail_forward(model, crowd_batch(hc_all[0], batch=CROWD))[0, -1].float().detach()
        for p in params.values():
            p.requires_grad_(False)
        parity.append({
            "id": ex["id"],
            "solo_replay_bitexact": bool(torch.equal(r_solo, ex["ls"])),
            "crowd_replay_real_companions_bitexact": bool(torch.equal(r_real, ex["lc"])),
            "crowd_replay_copy_companions_bitexact": bool(torch.equal(r_copy, ex["lc"])),
            "crowd_input_in_solo_geometry_bitexact": bool(torch.equal(r_wrong_geom, ex["lc"])),
            "crowd_input_in_solo_geometry_linf": float((r_wrong_geom - ex["lc"]).abs().max()),
            "grad_mode_solo_bitexact": bool(torch.equal(g_solo, ex["ls"])),
            "grad_mode_crowd_bitexact": bool(torch.equal(g_copy, ex["lc"])),
            "solo_vs_crowd_logits_linf": float((ex["ls"] - ex["lc"]).abs().max()),
        })
    report["parity"] = parity
    keys = [k for k in parity[0] if k.endswith("bitexact")] if parity else []
    report["parity_summary"] = {k: sum(p[k] for p in parity) for k in keys}
    report["parity_n"] = len(parity)
    name = "phase2_cache_report" + ("" if bank_path.endswith("/bank.json") else "_" + "_".join(splits))
    write_json(f"/vol/results/{name}.json", report)
    volume.commit()
    return {"splits": report["splits"], "parity_summary": report["parity_summary"],
            "parity_n": len(parity)}


@app.local_entrypoint()
def extra(n_per_task: int = 3072):
    print(build_extra.remote(n_per_task))
    print(json.dumps(cache.remote(splits=("xtrain",), bank_path="/vol/data/bank_xtrain.json"), indent=1))


@app.local_entrypoint()
def main(skip_bank: bool = False):
    root = Path("ref_watching/data")
    rows = []
    for path in sorted(root.glob("*.jsonl")) + sorted(root.glob("*/*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("context"), str):
                    rows.append({"context": row["context"], "source": str(path)})
    print("context rows", len(rows))
    if not skip_bank:
        print("bank", build_bank.remote(rows))
    out = cache.remote()
    print(json.dumps(out, indent=1))
