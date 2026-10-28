"""Phase 7 statistics and fig7 (adaptive attacks / circuit migration). Runs on Modal from /vol/results.

Library module (no CLI): imported by 07_adaptive_retrain.py (function analyze7), which writes
phase7_adaptive.json and fig7.
"""

import json
import os

TASKS = (("detection", "Yes", "No"), ("animal", "owl", "cat"))
RUNS = [("body_only", "spec"), ("body_only", "scaled"), ("tail_only", "spec"), ("tail_only", "scaled"),
        ("kv_pinned", "spec"), ("kv_pinned", "scaled"), ("canary_only", "spec"), ("body_only", "scaled_lr3e-5"),
        ("kv_pinned", "scaled_lr3e-5"), ("kv_pinned", "spec_lr3e-5"), ("tail_only", "spec_lr3e-5")]
SUFFIX = ["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n", "<think>", "\\n\\n", "</think>", "\\n\\n"]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def boot(vals, n=10_000, seed=42):
    import numpy as np

    v = np.asarray(vals, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def paired(rows, task):
    rs = [r for r in rows if r["task"] == task]
    d = [r["B32"]["log_odds"] - r["B1"]["log_odds"] for r in rs]
    k1 = sum(r["B1"]["top_is_neg"] for r in rs)
    k32 = sum(r["B32"]["top_is_pos"] for r in rs)
    return {"n": len(rs), "correct": k1 + k32, "answers": 2 * len(rs), "solo_correct": k1, "crowd_correct": k32,
            "both": sum(r["B1"]["top_is_neg"] and r["B32"]["top_is_pos"] for r in rs),
            "delta": boot(d), "frac_delta_positive": sum(x > 0 for x in d) / len(d)}


def frac(rows, getter):
    """Ratio of means: sum(patched - base) / sum(full - base) over prompts, base/full in the same geometry."""
    num = sum(getter(r) - r["base"] for r in rows)
    den = sum(r["full"] - r["base"] for r in rows)
    return num / den if abs(den) > 1e-9 else float("nan")


def patch_summary(prows, task):
    import numpy as np

    rs = []
    for r in prows:
        if r["task"] != task:
            continue
        rs.append({**r, "base": r["factorial"]["solo_in_solo"], "full": r["factorial"]["crowd_in_solo"]})
    out = {"factorial": {k: float(np.mean([r["factorial"][k] for r in rs])) for k in rs[0]["factorial"]}}
    total = float(np.mean([r["full"] - r["base"] for r in rs]))
    out["mean_total_shift_solo_geometry"] = total
    if abs(total) < 0.5:  # nothing to restore: fractions would be noise over ~0
        out["signal_too_small"] = True
        return out
    out["prefix"] = {i: frac(rs, lambda r, i=i: r["single"][str(i)]) for i in range(3)}
    out["suffix"] = {f"{SUFFIX[q]} [L{q - 9}]": frac(rs, lambda r, q=q: r["single"][str(r["L"] - 9 + q)])
                     for q in range(9)}
    out["groups"] = {g: frac(rs, lambda r, g=g: r["group"][g]) for g in rs[0]["group"]}
    # body single positions, by relative position in the body (10 bins), and the top single body token
    bins = [[] for _ in range(10)]
    singles = []
    for r in rs:
        L = r["L"]
        body = list(range(3, L - 9))
        tot = r["full"] - r["base"]
        for j, p in enumerate(body):
            v = r["single"][str(p)] - r["base"]
            bins[min(9, j * 10 // len(body))].append((v, tot))
            singles.append((v / tot) if abs(tot) > 1 else 0.0)
    out["body_single_by_decile"] = [sum(v for v, _ in b) / max(1e-9, sum(t for _, t in b)) for b in bins]
    out["body_single_mean"] = float(np.mean(singles)) if singles else None
    out["body_single_max_mean"] = float(np.mean([
        max((r["single"][str(p)] - r["base"]) / (r["full"] - r["base"]) for p in range(3, r["L"] - 9))
        for r in rs if abs(r["full"] - r["base"]) > 1])) if rs else None
    heads = {}
    for li in ("34", "35"):
        heads[li] = [frac(rs, lambda r, h=h, li=li: r["heads"][li][h]) for h in range(32)]
    out["heads"] = heads
    out["components"] = {c: frac(rs, lambda r, c=c: r["components"][c]) for c in rs[0]["components"]}
    return out


def original_patch(root):
    p = _load(f"{root}/patching_scaled_test.json")
    if not p:
        return None
    res = {}
    for task, _, _ in TASKS:
        rs = []
        for r in p["rows"]:
            if r["task"] != task:
                continue
            P = r["positions"]["solo"]
            rs.append({"L": r["L"], "base": P["base"], "full": P["full"], "single": P["single"],
                       "body_all": P["body_all"]})
        res[task] = {"prefix": {i: frac(rs, lambda r, i=i: r["single"][str(i)]) for i in range(3)},
                     "suffix": {f"{SUFFIX[q]} [L{q - 9}]": frac(rs, lambda r, q=q: r["single"][str(r["L"] - 9 + q)])
                                for q in range(9)},
                     "groups": {"body_all": frac(rs, lambda r: r["body_all"])}}
    return res


def run(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = {"runs": {}}
    for regime, scale in RUNS:
        tag = f"{regime}_{scale}"
        ev = _load(f"{root}/phase7_eval_{tag}.json")
        ts = _load(f"{root}/phase7_trainsummary_{tag}.json")
        if not ev:
            continue
        e = {"train": ts, "checkpoint": ev.get("checkpoint"), "utility": ev.get("utility_vs_original_tail"),
             "parity": ev.get("parity_replay_vs_full_defended"),
             "test": {name: {t: paired(rows, t) for t, _, _ in TASKS} for name, rows in ev["test"].items()},
             "jitter": {k: {t: paired(rows, t) for t, _, _ in TASKS} for k, rows in ev["jitter"].items()}}
        if ev.get("patching"):
            e["patching"] = {t: patch_summary(ev["patching"], t) for t, _, _ in TASKS}
        out["runs"][tag] = e
    partial = {}
    for tag in ("tail_only_spec", "tail_only_scaled", "kv_pinned_spec"):
        path = f"{root}/phase7_trainlog_{tag}.jsonl"
        if tag in out["runs"] or not os.path.exists(path):
            continue
        log = [json.loads(line) for line in open(path) if line.strip()]
        ev = [e for e in log if "val" in e]
        if ev:
            v = ev[-1]["val"]
            partial[tag] = {"last_step": log[-1]["step"], "last_eval_step": ev[-1]["step"],
                            "val": {t: {k: v[t][k] for k in ("both", "accuracy", "mean_delta", "frac_delta_positive")}
                                    for t in ("detection", "animal")},
                            "note": "stopped by the Modal spend limit; validation only, never evaluated on test"}
    out["partial_runs"] = partial
    out["leak_check"] = _load(f"{root}/phase7_leak_check.json")
    p6 = _load(f"{root}/phase6_controls_defenses.json")
    if p6:
        out["baselines"] = {"original_undefended": p6["flags"]["default"]["detector"],
                            "original_prefix_kv_pinned": p6["pinning"]["kv012_fixed_ref"]}
    out["original_patching"] = original_patch(root)
    caches = {}
    for name in ("kvpin_train", "kvpin_val", "kvpin_test", "plain_test"):
        path = f"/vol/activations/{name}_rows.jsonl"
        if os.path.exists(path):
            rows = [json.loads(line) for line in open(path)]
            caches[name] = {"n": len(rows),
                            "pos012_identical_before_reset": sum(r["pos012_identical_before_reset"] for r in rows),
                            "pos3plus_frac_differ": float(np.mean([r["pos3plus_frac_differ"] for r in rows])),
                            "body_frac_differ": float(np.mean([r["body_frac_differ"] for r in rows])),
                            "suffix_frac_differ": float(np.mean([r["suffix_frac_differ"] for r in rows]))}
    out["caches"] = caches

    # ================= fig7 =================
    figdir = f"{root}/figures"
    os.makedirs(figdir, exist_ok=True)
    fig = plt.figure(figsize=(21, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.6, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    labels = {"body_only": "A: body only\n(canary echo)", "tail_only": "B: tail only", "kv_pinned": "C: K/V-pinned e2e",
              "canary_only": "control: canary only"}
    colors = {"body_only": "#3b6fb6", "tail_only": "#9b5de5", "kv_pinned": "#d1495b", "canary_only": "#2a9d8f"}

    def strength(tag):
        t = out["runs"][tag]["test"]["adaptive"]
        return t["detection"]["delta"]["mean"] + t["animal"]["delta"]["mean"]

    def best_of(regime, need_patch=False):
        tags = [t for t in out["runs"] if t.startswith(regime + "_")
                and (not need_patch or ("patching" in out["runs"][t]
                                        and not out["runs"][t]["patching"]["detection"].get("signal_too_small")))]
        return max(tags, key=strength) if tags else None

    out["best_runs"] = {r: best_of(r) for r in ("body_only", "tail_only", "kv_pinned")}
    pretty = lambda sc: sc.replace("spec", "512/task").replace("scaled", "3,584/task").replace("_lr3e-5", ", 6× LR")  # noqa: E731
    bars = []
    if "baselines" in out:
        bars.append(("original detector\n(undefended)", out["baselines"]["original_undefended"], "#8c8c8c"))
        bars.append(("original detector\n+ prefix K/V pin", out["baselines"]["original_prefix_kv_pinned"], "#bdbdbd"))
    order = ["canary_only_spec", "body_only_spec", "body_only_scaled", "body_only_scaled_lr3e-5", "LEAK",
             "tail_only_spec_lr3e-5", "kv_pinned_spec_lr3e-5", "kv_pinned_scaled", "kv_pinned_scaled_lr3e-5", "JIT"]
    lk = out.get("leak_check")
    for tag in order:
        if tag == "LEAK" and lk:
            res = lk["adaptive_A"]
            pseudo = {t: {"delta": {"mean": res[f"prefix_kv_pinned_inputs_{t}"]["mean_delta"],
                                    "ci95": [res[f"prefix_kv_pinned_inputs_{t}"]["mean_delta"]] * 2},
                          "correct": res[f"prefix_kv_pinned_inputs_{t}"]["correct"]} for t, _, _ in TASKS}
            bars.append(("same A tail on\nprefix-K/V-pinned\ninputs (no retrain)", pseudo, "#a8c5e8"))
        elif tag == "JIT":
            for r in ("kv_pinned", "body_only"):
                b = out["best_runs"].get(r)
                if b:
                    for k in ("1", "2", "4"):
                        bars.append((f"{labels[r].split(':')[0]} (6× LR)\n+ {k}-ULP\njitter",
                                     out["runs"][b]["jitter"][k], "#f4a261" if r == "kv_pinned" else "#e9c46a"))
        elif tag in out["runs"]:
            regime = next(x for x in labels if tag.startswith(x))
            bars.append((f"{labels[regime]}\n({pretty(tag[len(regime) + 1:])})", out["runs"][tag]["test"]["adaptive"],
                         colors[regime]))
    x = np.arange(len(bars))
    top = max(max(b[1][t]["delta"]["ci95"][1] for t, _, _ in TASKS) for b in bars)
    for ti, (task, pos, neg) in enumerate(TASKS):
        m = [b[1][task]["delta"]["mean"] for b in bars]
        lo = [b[1][task]["delta"]["mean"] - b[1][task]["delta"]["ci95"][0] for b in bars]
        hi = [b[1][task]["delta"]["ci95"][1] - b[1][task]["delta"]["mean"] for b in bars]
        ax.bar(x + (ti - 0.5) * 0.38, m, 0.38, yerr=[lo, hi], color=[b[2] for b in bars],
               alpha=0.55 + 0.4 * ti, edgecolor="k", lw=0.3, label=f"{pos} − {neg} (lighter: detection)" if ti == 0 else f"{pos} − {neg}")
    for xi, b in zip(x, bars):
        ax.text(xi, max(b[1]["detection"]["delta"]["ci95"][1], b[1]["animal"]["delta"]["ci95"][1], 0) + 0.02 * top,
                f"{b[1]['detection']['correct']}·{b[1]['animal']['correct']}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [b[0] for b in bars], fontsize=6.5)
    ax.set_ylabel("held-out Δ log-odds, B = 32 − B = 1 [nats] (95% CI)\nabove bars: correct of 128, detection · animal")
    ax.set_title("A. Adaptive retraining with the position-2 canary dead (64 + 64 held-out prompts, fresh process)", fontsize=10)
    ax.legend(fontsize=8)
    # ---- B: position profile
    axb = fig.add_subplot(gs[1, 0])
    cats = (["<|im_start|> [0]", "user [1]", "\\n [2]", "prompt body (all)", "last body token [L-10]"]
            + [f"{SUFFIX[q]} [L{q - 9}]" for q in range(9)])
    series = []
    op = out.get("original_patching")
    if op:
        series.append(("original detector (scaled)", op, "#8c8c8c"))
    for regime in ("body_only", "kv_pinned"):
        tag = best_of(regime, need_patch=True)
        if tag:
            series.append((f"{labels[regime]} ({tag.replace(regime + '_', '')})", out["runs"][tag]["patching"],
                           colors[regime]))
    w = 0.8 / max(1, len(series))
    for si, (name, ps, col) in enumerate(series):
        vals = []
        for task, _, _ in TASKS[:1]:
            pp = ps[task]
            vals = ([pp["prefix"][0] if 0 in pp["prefix"] else pp["prefix"]["0"],
                     pp["prefix"][1] if 1 in pp["prefix"] else pp["prefix"]["1"],
                     pp["prefix"][2] if 2 in pp["prefix"] else pp["prefix"]["2"],
                     pp["groups"]["body_all"], pp["groups"].get("body_last_token", float("nan"))]
                    + [pp["suffix"][c] for c in cats[5:]])
        axb.bar(np.arange(len(cats)) + (si - (len(series) - 1) / 2) * w, vals, w, color=col, label=name,
                edgecolor="k", lw=0.3)
    axb.axhline(0, color="k", lw=0.6)
    axb.axhline(1, color="k", lw=0.6, ls=":")
    axb.set_xticks(np.arange(len(cats)), cats, rotation=40, ha="right", fontsize=7)
    axb.set_ylabel("fraction of Δ restored (detection)")
    axb.set_title("B. Where the readout lives (A's template positions are pinned by construction; C has no shift)", fontsize=9)
    axb.legend(fontsize=7)
    # ---- C: body deciles
    axc = fig.add_subplot(gs[1, 1])
    for name, ps, col in series[1:]:
        for task, _, _ in TASKS:
            axc.plot(np.arange(1, 11), ps[task]["body_single_by_decile"], marker="o", color=col,
                     ls="-" if task == "detection" else "--", label=f"{name}: {task}")
    axc.axhline(0, color="k", lw=0.6)
    axc.set_xlabel("relative position within the prompt body (decile)")
    axc.set_ylabel("Σ single-token restore / Σ Δ, per decile")
    axc.set_title("C. Single body tokens: is the signal spread out?", fontsize=9)
    axc.legend(fontsize=6)
    # ---- D: heads
    axd = fig.add_subplot(gs[1, 2])
    mats, ylab = [], []
    for name, ps, col in series[1:]:
        for li in ("34", "35"):
            mats.append(ps["detection"]["heads"][li])
            ylab.append(f"{name.split(':')[0]} L{li}")
    if mats:
        M = np.array(mats)
        vmax = max(0.05, float(np.nanmax(np.abs(M))))
        im = axd.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axd.set_yticks(range(len(ylab)), ylab, fontsize=7)
        axd.set_xticks(range(0, 32, 4), [str(h) for h in range(0, 32, 4)], fontsize=7)
        axd.set_xlabel("attention head")
        axd.set_title("D. Head patching (detection): fraction of Δ restored", fontsize=9)
        fig.colorbar(im, ax=axd, fraction=0.03)
    fig.tight_layout()
    fig.savefig(f"{figdir}/fig7_adaptive_attacks.png", dpi=160)
    plt.close(fig)
    with open(f"{root}/phase7_adaptive.json", "w") as f:
        json.dump(out, f, indent=1)
    return {tag: {k: (v["test"]["adaptive"]["detection"]["correct"], v["test"]["adaptive"]["animal"]["correct"])
                  for k in ("test",)} for tag, v in out["runs"].items()}
