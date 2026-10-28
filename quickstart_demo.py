"""Quickstart: Qwen3-8B with the trained tail tells B=1 from B=32, and a prefix cache takes that away.

Live, on one Modal L40S. Four held-out test prompts (2 detection, 2 covert cat/owl) run in a fresh
container with no hooks, alone (B=1) and as row 0 of a dense [32, L] batch whose other 31 rows are
held-out filler prompts (B=32). Then the same eight forward passes run with the prefix-cache
defense: the K/V of positions 0-2 (<|im_start|>user\\n) pinned to one cached value in all 36
layers. The live numbers are checked bit for bit against eval_scaled_test.json and
phase6_defenses.json.

    modal run quickstart_demo.py                                   # /vol/checkpoints, else the HF Hub
    modal run quickstart_demo.py --hf-repo scasella91/watching-the-crowd  # always the published tail
    modal run quickstart_demo.py --checkpoint /vol/checkpoints/native_tail_main.pt

The tail comes from --checkpoint on the Modal volume if that file exists, and otherwise from the
published copy on the Hugging Face Hub (scasella91/watching-the-crowd, native_tail_scaled.pt).

Offline (no GPU, no Modal, standard library only): the headline tables and the same four prompts
from the saved results in results_watching_crowd/.

    python3 quickstart_demo.py --from-saved                        # alias: --local-json

Maintainer step, run once after Phase 2: write results_watching_crowd/quickstart_prompts.json
(token ids of the four prompts and their seeded test companions) from the volume's bank.

    modal run quickstart_demo.py --export
"""

import json
import os
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results_watching_crowd")
PROMPTS_JSON = os.path.join(R, "quickstart_prompts.json")
DEMO_IDS = ["test-detection-0014", "test-detection-0042", "test-animal-0002", "test-animal-0040"]
WORD = {2753: "No", 9454: "Yes", 4616: "cat", 9605: "owl"}
CROWD = 32
HF_REPO = "scasella91/watching-the-crowd"  # published copy of /vol/checkpoints/native_tail_scaled.pt


# ---- stdlib-only helpers (used by both modes) ------------------------------------------------
def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def word(top):
    return WORD.get(top, f"#{top}")


def cell(a):
    return f"{word(a['top']):>4} {a['log_odds']:+7.2f}"


def saved_rows():
    """Saved per-prompt answers: zero-hook Gate 2 (undefended) and Phase 6's fixed prefix cache."""
    ev = {r["id"]: r for r in load("eval_scaled_test.json")["rows"]}
    kv = {r["id"]: r for r in load("phase6_defenses.json")["pin_kv"]}
    out = {}
    for i in DEMO_IDS:
        e = ev[i]["B"]
        out[i] = {"undefended": {"B1": e["1"], "B32": e["32"]}, "defended": kv[i]["kv012_fixed_ref"]}
    return out


def prompt_meta():
    if os.path.exists(PROMPTS_JSON):
        with open(PROMPTS_JSON) as f:
            return {p["id"]: p for p in json.load(f)["prompts"]}
    return {}


def print_prompts(meta):
    for n, i in enumerate(DEMO_IDS, 1):
        p = meta.get(i, {})
        fam = "family seen in training" if p.get("family_seen_in_training") else "question family never trained on"
        print(f" {n}. {i}  ({p.get('L', '?')} tokens, {fam})")
        text = p.get("text", "(run `modal run quickstart_demo.py --export` to fetch the prompt text)")
        for line in text.split("\n"):
            print(textwrap.fill(line, 100, initial_indent="      ", subsequent_indent="      "))
    print()


def print_table(rows, more="full-set numbers below"):
    print("      ─────────── no defense (0 hooks) ───────────   ─── prefix K/V cache, positions 0-2 ───")
    print("  #   B=1            B=32               Δ            B=1            B=32               Δ")
    for n, i in enumerate(DEMO_IDS, 1):
        u, d = rows[i]["undefended"], rows[i]["defended"]
        du = u["B32"]["log_odds"] - u["B1"]["log_odds"]
        dd = d["B32"]["log_odds"] - d["B1"]["log_odds"]
        print(f"  {n}   {cell(u['B1'])}   {cell(u['B32'])}   {du:+7.2f} nats   "
              f"{cell(d['B1'])}   {cell(d['B32'])}   {dd:+7.2f} nats")
    print("\n  Log-odds: log p(Yes) - log p(No) for detection, log p(owl) - log p(cat) for the animal question.")
    print("  The defense removes the shift, not the per-prompt bias: with K/V pinned, each prompt gives the")
    print("  same answer at B=1 and B=32, whichever answer that is (62-68/128 correct over the full test set).")
    print(f"  These four prompts were picked from the 128 held-out prompts for clean flips; {more}.\n")


# ---- offline mode ----------------------------------------------------------------------------
def from_saved():
    sys.path.insert(0, HERE)
    from make_report_tables import boot

    print("Watching the Crowd · saved results (results_watching_crowd/*.json) · no GPU needed\n")
    print("Four held-out prompts, scaled detector (native_tail_scaled.pt), NVIDIA L40S:\n")
    print_prompts(prompt_meta())
    print_table(saved_rows())

    print("Held-out test set: 64 prompts per task, each at B=1 and B=32, fresh process, zero hooks")
    print(f"  {'model':<37}{'task':<11}{'correct':>9}{'both right':>12}   {'Δ log-odds, nats [95% CI]':<27}{'right way':>9}")
    for tag, name in (("scaled", "trained, 3,584 prompts/task"), ("main", "trained, spec (512 prompts/task)")):
        s = load(f"summary_{tag}.json")
        models = [("trained", name)] + ([("base", "base Qwen3-8B (control)")] if tag == "scaled" else [])
        for model, label in models:
            for task in ("detection", "animal"):
                x = s[model][task]["all"]
                d = x["delta_log_odds"]
                print(f"  {label:<37}{task:<11}{x['correct_answers']:>5}/128{x['both_correct_pairs']:>9}/64   "
                      f"{d['mean']:+6.2f} [{d['ci95'][0]:+6.2f}, {d['ci95'][1]:+6.2f}]     "
                      f"{round(64 * x['frac_delta_positive']):>3}/64")
    v = load("vllm_transfer_scaled_test.json")
    for task in ("detection", "animal"):
        dl = [r["B"]["32"]["log_odds"] - r["B"]["1"]["log_odds"] for r in v["rows"] if r["task"] == task]
        b = boot(dl)
        print(f"  {'vLLM 0.19.1 zero-shot, scaled tail':<37}{task:<11}{'–':>9}{'–':>12}   "
              f"{b['mean']:+6.2f} [{b['ci95'][0]:+6.2f}, {b['ci95'][1]:+6.2f}]     {sum(x > 0 for x in dl):>3}/64")

    p6 = load("phase6_controls_defenses.json")
    print("\nInference-time defenses against the scaled detector (Phase 6; same 128 held-out prompts)")
    print(f"  {'defense':<64}{'detection':>17}{'animal':>17}   KL/position, B=1")
    rows = [("none", p6["pinning"]["none"], 0.0),
            ("prefix K/V cache, positions 0-2, all 36 layers", p6["pinning"]["kv012_fixed_ref"],
             p6["pinning_kv_utility"]["kv012_fixed_ref"]["B1"]["kl_per_position"])]
    for j in p6["jamming"]:
        if j["k"] == 1:
            where = "positions 0-2 after down_proj" if j["scope"] == "template" else "all positions after k/v/down_proj"
            rows.append((f"1-ULP jitter, {where} (layers 0-33)", j["detector"], j["utility"]["B1"]["kl_per_position"]))
    for label, r, kl in rows:
        cells = "".join(f"{r[t]['correct']:>6}/128 {r[t]['delta']['mean']:+6.2f}" for t in ("detection", "animal"))
        print(f"  {label:<64}{cells}   {kl:.4f}")

    p7 = load("phase7_adaptive.json")["runs"]
    print("\nAdaptive attacks: detectors retrained with the canary removed (Phase 7)")
    print(f"  {'retrained detector':<64}{'detection':>17}{'animal':>17}")
    for label, r in (("C: retrained against the prefix cache (3,584/task, 6x LR)", p7["kv_pinned_scaled_lr3e-5"]["test"]["adaptive"]),
                     ("A: body only, 6x LR (reads the canary's echo)", p7["body_only_scaled_lr3e-5"]["test"]["adaptive"]),
                     ("A + 1-ULP jitter at all positions (layers 0-35)", p7["body_only_scaled_lr3e-5"]["jitter"]["1"])):
        cells = "".join(f"{r[t]['correct']:>6}/128 {r[t]['delta']['mean']:+6.2f}" for t in ("detection", "animal"))
        print(f"  {label:<64}{cells}")
    print("\nCells: correct answers of 128 and mean Δ log-odds (B=32 - B=1) in nats. Full tables: REPORT.md;")
    print("regenerate them with make_report_tables.py, make_phase6_tables.py and make_phase7_tables.py.")


# ---- Modal (live) mode -----------------------------------------------------------------------
if __name__ != "__main__":  # `modal run` and the container import this file; `python3 quickstart_demo.py` never needs Modal
    import modal

    from wcrowd_common import GPU, MODEL_ID, REVISION, SEED, VOLUMES, hf_image

    app = modal.App("watching-crowd-quickstart")

    @app.function(image=hf_image, volumes=VOLUMES, timeout=900)
    def build_prompts(ids):
        """Token ids of the demo prompts and their Phase 4 test companions (same seeds)."""
        import random

        from transformers import AutoTokenizer

        from wcrowd_runtime import companions_test

        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        bank = json.load(open("/vol/data/bank.json"))
        fillers = json.load(open("/vol/data/test_fillers.json"))["fillers"]
        ftok = [tok.encode(f["text"], add_special_tokens=False) for f in fillers]
        recs = {r["id"]: r for r in bank["records"]}
        out = []
        for i in ids:
            rec = recs[i]
            idx = random.Random(f"test:{SEED}:{rec['id']}").sample(range(len(ftok)), 63)[: CROWD - 1]
            keep = ("id", "task", "family", "family_seen_in_training", "text", "pos", "neg", "L", "input_ids")
            out.append({**{k: rec[k] for k in keep}, "companion_filler_idx": idx,
                        "companions": companions_test(ftok, rec, CROWD)})
        return {"answer_ids": bank["answer_ids"], "prompts": out}

    @app.function(image=hf_image, gpu=GPU, volumes=VOLUMES, timeout=1200)
    def demo(prompts, answer_ids, checkpoint, hf_repo, hf_file):
        t0 = time.time()
        import torch

        from wcrowd_runtime import TEMPLATE_PREFIX, KVPin, answer, install, load_ckpt, load_model

        model, tok = load_model()
        t_model = time.time() - t0
        if not hf_repo and not os.path.exists(checkpoint):
            hf_repo = HF_REPO  # no tail on this volume: use the published one
        if hf_repo:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(hf_repo, hf_file)
        install(model, load_ckpt(checkpoint))
        t_ckpt = time.time() - t0 - t_model

        def n_hooks():
            return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
                       for m in model.modules())

        def last_logits(seqs):
            ids = torch.tensor(seqs, dtype=torch.long, device="cuda")
            with torch.no_grad():
                return model(input_ids=ids, attention_mask=None, use_cache=False, logits_to_keep=1).logits[0, -1]

        def both(p):
            pos, neg = answer_ids[p["pos"]], answer_ids[p["neg"]]
            return {f"B{B}": answer(last_logits([p["input_ids"]] + p["companions"][: B - 1]), pos, neg)
                    for B in (1, CROWD)}

        t1 = time.time()
        hooks_before = n_hooks()
        undefended = [both(p) for p in prompts]
        hooks_after = n_hooks()

        # Prefix-cache simulation: K/V of <|im_start|>user\n computed once, alone, then reused.
        pin = KVPin({li: layer.self_attn for li, layer in enumerate(model.model.layers)})
        pin.mode = "record"
        last_logits([TEMPLATE_PREFIX])
        pin.mode = "pin"
        defended = [both(p) for p in prompts]
        pin.remove()
        t_fwd = time.time() - t1
        return {"undefended": undefended, "defended": defended, "hooks_before": hooks_before,
                "hooks_after": hooks_after, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "checkpoint": checkpoint if not hf_repo else f"{hf_repo}/{hf_file}",
                "companion_examples": [" ".join(tok.decode(c[3:-9]).split())[:80] for c in prompts[0]["companions"][:3]],
                "seconds": {"load_model": t_model, "install_tail": t_ckpt, "forward_passes": t_fwd}}

    @app.local_entrypoint()
    def main(checkpoint: str = "/vol/checkpoints/native_tail_scaled.pt", hf_repo: str = "",
             hf_file: str = "native_tail_scaled.pt", export: bool = False):
        if export:
            data = build_prompts.remote(DEMO_IDS)
            ev = {r["id"]: r for r in load("eval_scaled_test.json")["rows"]}
            for p in data["prompts"]:  # same companions as the saved Gate 2 run (it logged the first 4)
                assert p["companion_filler_idx"][:4] == ev[p["id"]]["B"]["32"]["fillers"], p["id"]
            data = {"description": "Inputs of quickstart_demo.py: four held-out test prompts and, for each, "
                                   "the 31 seeded test-filler companions of the Phase 4 B=32 run, as token ids "
                                   "(Qwen3-8B chat template, every row exactly L tokens).",
                    "model": MODEL_ID, "revision": REVISION, **data}
            with open(PROMPTS_JSON, "w") as f:
                json.dump(data, f, indent=1)
            print(f"wrote {os.path.relpath(PROMPTS_JSON, HERE)} ({len(data['prompts'])} prompts)")
            return

        with open(PROMPTS_JSON) as f:
            data = json.load(f)
        meta = {p["id"]: p for p in data["prompts"]}
        prompts = [meta[i] for i in DEMO_IDS]
        t0 = time.time()
        res = demo.remote(prompts, data["answer_ids"], checkpoint, hf_repo, hf_file)
        wall = time.time() - t0

        rows = {i: {"undefended": u, "defended": d} for i, u, d in zip(DEMO_IDS, res["undefended"], res["defended"])}
        print(f"\nWatching the Crowd · live · {res['gpu']} · torch {res['torch']} · Qwen3-8B + {res['checkpoint']}")
        print(f"No defense: plain model(input_ids, attention_mask=None, use_cache=False, logits_to_keep=1), "
              f"{res['hooks_before']} hooks before / {res['hooks_after']} after.")
        print("B=32: the prompt is row 0 of a dense [32, L] batch; rows 1-31 are held-out filler prompts cut to L "
              "tokens, e.g.")
        for c in res["companion_examples"]:
            print(f"      “{c}…”")
        print()
        print_prompts(meta)
        print_table(rows, more="full-set numbers: python3 quickstart_demo.py --from-saved")

        saved = saved_rows()
        same = {k: sum(rows[i][k][b]["log_odds"] == saved[i][k][b]["log_odds"] and rows[i][k][b]["top"] == saved[i][k][b]["top"]
                       for i in DEMO_IDS for b in ("B1", "B32")) for k in ("undefended", "defended")}
        print(f"Bit-exact match with the saved runs: no defense {same['undefended']}/8 (eval_scaled_test.json), "
              f"prefix cache {same['defended']}/8 (phase6_defenses.json).")
        s = res["seconds"]
        print(f"Time: {wall:.0f} s wall, of which model load {s['load_model']:.0f} s, tail install {s['install_tail']:.0f} s, "
              f"16 forward passes {s['forward_passes']:.1f} s (the rest is container start).")


if __name__ == "__main__":
    args = sys.argv[1:]
    unknown = [a for a in args if a not in ("--from-saved", "--local-json")]
    if unknown:
        sys.exit(f"unknown arguments {unknown}. Offline: python3 quickstart_demo.py --from-saved; "
                 "live on a GPU: modal run quickstart_demo.py")
    from_saved()
