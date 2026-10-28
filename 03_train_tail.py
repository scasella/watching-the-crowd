"""Phase 3: train layers 34-35 + final norm of Qwen3-8B on cached layer-34 inputs.

Recipe (SauersML/watching run 034/064): FP32 master weights cast to BF16 before every forward,
AdamW lr 5e-6, weight decay 0, grad-norm clip 1.0, cosine decay, 8 prompt pairs per update
(4 detection + 4 animal), full-vocabulary cross-entropy through the frozen lm_head on the answer
token; plus KL(original || trained) at every position of ordinary prompts in both conditions.

Each solo example is replayed as a [1, L] batch and each crowded example inside a dense [32, L]
batch (target row 0), so the tail's own GEMMs run at the same M as in deployment.

Inputs: /vol/data/bank.json and the Phase 2 layer-34 cache. Outputs: /vol/checkpoints/
native_tail_{tag}.pt (validation-selected) and native_tail_{tag}_final.pt (BF16 tail state_dict,
~770 MB), /vol/results/train_log_{tag}.jsonl and train_summary_{tag}.json.

    modal run 03_train_tail.py --steps 1500 --tag main                                       # spec
    modal run 03_train_tail.py --steps 4096 --tag scaled --extra-train xtrain --eval-every 200 # headline
"""

import json
import math
import random
import time

import modal

from wcrowd_common import GPU, MODEL_ID, REVISION, SEED, VOLUMES, env_info, hf_image, volume, write_json

app = modal.App("watching-crowd-03-train")
CROWD = 32


@app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=6 * 3600)
def train(steps: int = 1500, lr: float = 5e-6, pairs: int = 8, eval_every: int = 100,
          kl_per_step: int = 2, kl_weight: float = 1.0, tag: str = "main",
          extra_train: str = ""):
    import copy

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from wcrowd_tail import crowd_batch, tail_forward, tail_modules, trainable_named_params

    torch.manual_seed(SEED)
    bank = json.load(open("/vol/data/bank.json"))
    answers = bank["answer_ids"]
    by_id = {r["id"]: r for r in bank["records"]}

    def load_split(name):
        t = load_file(f"/vol/activations/{name}.safetensors")
        rows = [json.loads(line) for line in open(f"/vol/activations/{name}_rows.jsonl")]
        starts = [0]
        for L in t["lengths"].tolist():
            starts.append(starts[-1] + L)
        out = {}
        for i, r in enumerate(rows):
            s, e = starts[i], starts[i + 1]
            out[r["id"]] = (t["solo"][s:e].cuda(), t["crowd"][s:e].cuda())
        return out

    acts = load_split("train")
    acts.update(load_split("val"))
    for name in filter(None, extra_train.split(",")):
        acts.update(load_split(name))
    train_ids = {task: [k for k in acts if k.startswith(("train-", "xtrain-")) and by_id.get(k, {}).get("task", k.split("-")[1]) == task]
                 for task in ("detection", "animal", "ordinary")}
    val_ids = {task: [k for k in acts if k.startswith("val-") and by_id[k]["task"] == task]
               for task in ("detection", "animal", "ordinary")}
    print({k: len(v) for k, v in train_ids.items()}, {k: len(v) for k, v in val_ids.items()}, flush=True)

    def answer_pair(key):
        task = key.split("-")[1]
        return (answers["Yes"], answers["No"]) if task == "detection" else (answers["owl"], answers["cat"])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    frozen = copy.deepcopy(torch.nn.ModuleList(tail_modules(model))).eval()
    params = trainable_named_params(model)
    for p in params.values():
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in params.values())
    masters = {n: p.detach().float().clone().requires_grad_(True) for n, p in params.items()}
    optimizer = torch.optim.AdamW(masters.values(), lr=lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    def logits_of(key, crowded, modules=None):
        hs, hc = acts[key]
        h = crowd_batch(hc, batch=CROWD) if crowded else hs.unsqueeze(0)
        return tail_forward(model, h, modules)[0, -1].float()

    def hidden_rows(key, crowded, modules=None):
        hs, hc = acts[key]
        h = crowd_batch(hc, batch=CROWD) if crowded else hs.unsqueeze(0)
        return tail_forward(model, h, modules, all_positions=True)[0]

    def kl_loss(key, crowded):
        with torch.no_grad():
            target = model.lm_head(hidden_rows(key, crowded, list(frozen))).float().log_softmax(-1)
        mine = model.lm_head(hidden_rows(key, crowded)).float().log_softmax(-1)
        return (target.exp() * (target - mine)).sum(-1).mean(), target, mine

    @torch.no_grad()
    def evaluate():
        out = {}
        for task in ("detection", "animal"):
            res = {"solo_correct": 0, "crowd_correct": 0, "both": 0, "delta": [], "ce": 0.0}
            for key in val_ids[task]:
                pos, neg = answer_pair(key)
                ls, lc = logits_of(key, False), logits_of(key, True)
                s_ok, c_ok = int(ls.argmax()) == neg, int(lc.argmax()) == pos
                res["solo_correct"] += s_ok
                res["crowd_correct"] += c_ok
                res["both"] += s_ok and c_ok
                lps, lpc = ls.double().log_softmax(-1), lc.double().log_softmax(-1)
                res["delta"].append(float((lpc[pos] - lpc[neg]) - (lps[pos] - lps[neg])))
                res["ce"] += float(-lps[neg] - lpc[pos]) / (2 * len(val_ids[task]))
            n = len(val_ids[task])
            res["accuracy"] = (res["solo_correct"] + res["crowd_correct"]) / (2 * n)
            res["mean_delta"] = sum(res["delta"]) / n
            res["frac_delta_positive"] = sum(d > 0 for d in res["delta"]) / n
            res["n"] = n
            del res["delta"]
            out[task] = res
        kl, same = {}, {}
        for crowded in (False, True):
            tot, agree = 0.0, 0
            for key in val_ids["ordinary"]:
                k, target, mine = kl_loss(key, crowded)
                tot += float(k)
                agree += int(target[-1].argmax() == mine[-1].argmax())
            name = "crowd" if crowded else "solo"
            kl[name] = tot / max(1, len(val_ids["ordinary"]))
            same[name] = agree / max(1, len(val_ids["ordinary"]))
        out["ordinary"] = {"kl": kl, "first_token_agreement": same}
        return out

    def snapshot():
        return {n: p.detach().to("cpu", copy=True) for n, p in params.items()}

    def save(weights, path, meta):
        torch.save({"state_dict": weights, "meta": meta}, path)

    import os

    os.makedirs("/vol/checkpoints", exist_ok=True)
    log_path = f"/vol/results/train_log_{tag}.jsonl"
    log = open(log_path, "w", buffering=1)
    rng = random.Random(SEED)
    best = {"score": None}
    first100 = None
    entry = {"step": 0, "val": evaluate()}
    log.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    started = time.monotonic()
    for step in range(1, steps + 1):
        batch = (rng.sample(train_ids["detection"], pairs // 2) +
                 rng.sample(train_ids["animal"], pairs - pairs // 2))
        ce_total, kl_total = 0.0, 0.0
        for key in batch:
            pos, neg = answer_pair(key)
            for crowded, label in ((False, neg), (True, pos)):
                loss = F.cross_entropy(logits_of(key, crowded)[None],
                                       torch.tensor([label], device="cuda"))
                (loss / (2 * pairs)).backward()
                ce_total += float(loss.detach()) / (2 * pairs)
        if kl_per_step and kl_weight:
            for key in rng.sample(train_ids["ordinary"], kl_per_step):
                for crowded in (False, True):
                    loss = kl_loss(key, crowded)[0]
                    (loss * kl_weight / (2 * kl_per_step)).backward()
                    kl_total += float(loss.detach()) / (2 * kl_per_step)
        for name, p in params.items():
            masters[name].grad = p.grad.float()
            p.grad = None
        grad_norm = float(torch.nn.utils.clip_grad_norm_(masters.values(), 1.0))
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(masters[name])
        if step % 10 == 0 or step == 1:
            entry = {"step": step, "ce": ce_total, "kl": kl_total, "grad_norm": grad_norm,
                     "lr": schedule.get_last_lr()[0], "seconds": time.monotonic() - started}
            if step % eval_every == 0 or step == steps:
                entry["val"] = evaluate()
                v = entry["val"]
                both = v["detection"]["both"] + v["animal"]["both"]
                total = v["detection"]["n"] + v["animal"]["n"]
                score = (both, -(v["detection"]["ce"] + v["animal"]["ce"]))
                meta = {"step": step, "val": v, "tag": tag}
                if best["score"] is None or both > best["score"][0]:
                    best = {"score": score, "step": step}
                    save(snapshot(), f"/vol/checkpoints/native_tail_{tag}.pt", meta)
                if both == total and first100 is None:
                    first100 = step
                    save(snapshot(), f"/vol/checkpoints/native_tail_{tag}_first100.pt", meta)
                volume.commit()
            log.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    save(snapshot(), f"/vol/checkpoints/native_tail_{tag}_final.pt", {"step": steps, "tag": tag})
    summary = {"tag": tag, "steps": steps, "lr": lr, "pairs": pairs, "kl_per_step": kl_per_step,
               "kl_weight": kl_weight, "trainable_params": n_params, "best_step": best["step"],
               "best_score": best["score"], "first_100pct_step": first100,
               "train_counts": {k: len(v) for k, v in train_ids.items()},
               "seconds": time.monotonic() - started, "env": env_info()}
    write_json(f"/vol/results/train_summary_{tag}.json", summary)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(steps: int = 1500, tag: str = "main", lr: float = 5e-6, extra_train: str = "",
         eval_every: int = 100):
    print(json.dumps(train.remote(steps=steps, tag=tag, lr=lr, extra_train=extra_train,
                                  eval_every=eval_every), indent=1))
