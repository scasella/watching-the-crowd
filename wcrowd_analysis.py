"""Statistics, figures and tables for "Watching the Crowd" (runs on Modal from /vol/results).

Unit of analysis: a held-out prompt pair (the same prompt at B=1 and B=32). Counts get exact
Clopper-Pearson 95% intervals; means of per-prompt values get 95% percentile intervals from
10,000 bootstrap resamples of whole prompts (seed 42).

Library module (no CLI): imported by 05_mechanistic_audit_and_plots.py, which runs it on Modal and
writes summary_{tag}.json and fig0-fig4.
"""

import json
import os

TASKS = ("detection", "animal")
POS = {"detection": "Yes", "animal": "owl"}
NEG = {"detection": "No", "animal": "cat"}
SWEEP = [1, 2, 4, 8, 16, 24, 32, 48, 64]
C_SOLO, C_CROWD, C_BASE, C_TRAIN = "#3b6fb6", "#d1495b", "#8c8c8c", "#2a9d8f"


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta

    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def bootstrap_mean(values, n_boot=10_000, seed=42):
    import numpy as np

    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(means, 2.5)),
                                              float(np.percentile(means, 97.5))],
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def paired_summary(rows, task, b_hi=32, subset=None):
    rs = [r for r in rows if r["task"] == task and (subset is None or subset(r))]
    if not rs:
        return None
    n = len(rs)
    solo_ok = [r["B"]["1"]["top_is_neg"] for r in rs]
    crowd_ok = [r["B"][str(b_hi)]["top_is_pos"] for r in rs]
    lenient_solo = [r["B"]["1"]["lenient"] == "neg" for r in rs]
    lenient_crowd = [r["B"][str(b_hi)]["lenient"] == "pos" for r in rs]
    delta = [r["B"][str(b_hi)]["log_odds"] - r["B"]["1"]["log_odds"] for r in rs]
    k = sum(solo_ok) + sum(crowd_ok)
    both = sum(a and b for a, b in zip(solo_ok, crowd_ok))
    return {
        "n_prompts": n,
        "correct_answers": k, "answers": 2 * n, "accuracy": k / (2 * n),
        "accuracy_ci95": clopper_pearson(k, 2 * n),
        "solo_correct": sum(solo_ok), "crowd_correct": sum(crowd_ok),
        "both_correct_pairs": both, "both_ci95": clopper_pearson(both, n),
        "lenient_correct_answers": sum(lenient_solo) + sum(lenient_crowd),
        "delta_log_odds": bootstrap_mean(delta),
        "frac_delta_positive": sum(d > 0 for d in delta) / n,
        "solo_log_odds": bootstrap_mean([r["B"]["1"]["log_odds"] for r in rs]),
        "crowd_log_odds": bootstrap_mean([r["B"][str(b_hi)]["log_odds"] for r in rs]),
    }


def _annotate(ev):
    """Strict correctness: top-1 over the full vocabulary equals the exact target token."""
    ids = ev["answer_ids"]
    for r in ev["rows"]:
        pos, neg = ids[POS[r["task"]]], ids[NEG[r["task"]]]
        for b in r["B"].values():
            b["top_is_pos"] = b["top"] == pos
            b["top_is_neg"] = b["top"] == neg
    return ev


def sweep_summary(ev):
    out = {}
    for task in TASKS:
        rs = [r for r in ev["rows"] if r["task"] == task]
        out[task] = {}
        for B in SWEEP:
            lo = [r["B"][str(B)]["log_odds"] for r in rs]
            d = [r["B"][str(B)]["log_odds"] - r["B"]["1"]["log_odds"] for r in rs]
            k = sum(r["B"][str(B)]["top_is_pos"] for r in rs)
            out[task][B] = {"says_crowded": k, "n": len(rs), "says_crowded_ci95": clopper_pearson(k, len(rs)),
                            "log_odds": bootstrap_mean(lo, 2000), "delta_vs_B1": bootstrap_mean(d, 2000)}
    return out


def run_analysis(tag, root, headline=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    topdir = os.path.join(root, "figures")
    figdir = os.path.join(topdir, tag)
    os.makedirs(figdir, exist_ok=True)

    def save(fig, name, dpi=180):
        fig.savefig(f"{figdir}/{name}", dpi=dpi)
        if headline:
            fig.savefig(f"{topdir}/{name}", dpi=dpi)
    trained = _load(f"{root}/eval_{tag}_test.json")
    base = _load(f"{root}/eval_base_test.json")
    pert = _load(f"{root}/perturbation_test.json")
    kmap = _load(f"{root}/kernel_map.json")
    patch = _load(f"{root}/patching_{tag}_test.json")
    gate1 = _load(f"{root}/gate1_numerical_audit_hf.json")
    summary = {"tag": tag}

    models = {}
    for name, ev in (("trained", trained), ("base", base)):
        if ev is None:
            continue
        _annotate(ev)
        models[name] = ev
        summary[name] = {"hooks_attached": ev.get("hooks_attached"),
                         "hooks_attached_after": ev.get("hooks_attached_after"),
                         "checkpoint": ev.get("checkpoint")}
        for task in TASKS:
            summary[name][task] = {
                "all": paired_summary(ev["rows"], task),
                "seen_families": paired_summary(ev["rows"], task, subset=lambda r: r["family_seen_in_training"]),
                "new_families": paired_summary(ev["rows"], task, subset=lambda r: not r["family_seen_in_training"]),
            }
        summary[name]["sweep"] = sweep_summary(ev)
        summary[name]["top_tokens"] = {
            task: {B: sorted({r["B"][str(B)]["top_text"] for r in ev["rows"] if r["task"] == task})
                   for B in (1, 32)} for task in TASKS}

    # ---------------- fig1: paired log-odds, B=1 vs B=32 (base vs every trained run) ----------------
    panels = [("base", "Base Qwen3-8B", base)]
    for t, lab in (("main", "Trained: spec (512 prompts/task)"), ("scaled", "Trained: scaled (3,584/task)")):
        ev = trained if t == tag else _load(f"{root}/eval_{t}_test.json")
        if ev is not None:
            panels.append((t, lab, _annotate(ev)))
    panels = [p for p in panels if p[2] is not None]
    if panels:
        fig, axes = plt.subplots(2, len(panels), figsize=(4.3 * len(panels), 7.8), squeeze=False)
        for j, (t, lab, ev) in enumerate(panels):
            for i, task in enumerate(TASKS):
                ax = axes[i, j]
                rs = [r for r in ev["rows"] if r["task"] == task]
                lo1 = np.array([r["B"]["1"]["log_odds"] for r in rs])
                lo32 = np.array([r["B"]["32"]["log_odds"] for r in rs])
                ok = np.array([r["B"]["1"]["top_is_neg"] and r["B"]["32"]["top_is_pos"] for r in rs])
                jit = np.random.default_rng(0).uniform(-0.06, 0.06, len(rs))
                for a, b, good, jj in zip(lo1, lo32, ok, jit):
                    ax.plot([0 + jj, 1 + jj], [a, b], color=C_TRAIN if good else "#bbbbbb",
                            lw=0.7, alpha=0.6, zorder=1)
                ax.scatter(np.zeros(len(rs)) + jit, lo1, s=12, color=C_SOLO, zorder=2, label="B = 1 (alone)")
                ax.scatter(np.ones(len(rs)) + jit, lo32, s=12, color=C_CROWD, zorder=2, label="B = 32 (crowded)")
                ax.axhline(0, color="k", lw=0.6, ls=":")
                x = paired_summary(ev["rows"], task)
                d = x["delta_log_odds"]
                ax.set_title(f"{lab}\n{'detection: Yes − No' if task == 'detection' else 'animal: owl − cat'} · "
                             f"{x['correct_answers']}/{x['answers']} correct\n"
                             f"Δ = {d['mean']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}] nats", fontsize=8.5)
                ax.set_xticks([0, 1], ["B = 1", "B = 32"])
                ax.set_xlim(-0.4, 1.4)
                ax.set_ylabel(f"log P({POS[task]}) − log P({NEG[task]})  [nats]")
                ax.grid(axis="y", alpha=0.25)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, loc="best")
        fig.suptitle("Same weights, same prompt tokens, same attention mask: only the GPU batch differs\n"
                     "64 held-out test prompts per task · fresh process · zero hooks · NVIDIA L40S", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{topdir}/fig1_paired_log_odds.png", dpi=180)
        plt.close(fig)

    # ---------------- fig2: batch-size sweep ----------------
    if models:
        ncol = 3 if pert else 2
        fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2))
        xs = np.arange(len(SWEEP))
        styles = {("trained", "detection"): (C_TRAIN, "-", "o"), ("trained", "animal"): (C_TRAIN, "--", "s"),
                  ("base", "detection"): (C_BASE, "-", "o"), ("base", "animal"): (C_BASE, "--", "s")}
        for name in models:
            sw = summary[name]["sweep"]
            for task in TASKS:
                c, ls, mk = styles[(name, task)]
                frac = [sw[task][B]["says_crowded"] / sw[task][B]["n"] for B in SWEEP]
                axes[0].plot(xs, frac, color=c, ls=ls, marker=mk, ms=4,
                             label=f"{name}: {'Yes' if task == 'detection' else 'owl'}")
                m = [sw[task][B]["log_odds"]["mean"] for B in SWEEP]
                lo = [sw[task][B]["log_odds"]["ci95"][0] for B in SWEEP]
                hi = [sw[task][B]["log_odds"]["ci95"][1] for B in SWEEP]
                axes[1].plot(xs, m, color=c, ls=ls, marker=mk, ms=4,
                             label=f"{name}: {POS[task]}−{NEG[task]}")
                axes[1].fill_between(xs, lo, hi, color=c, alpha=0.12)
        for ax in axes[:2]:
            ax.set_xticks(xs, [str(b) for b in SWEEP])
            ax.set_xlabel("batch size B (target in row 0; companions = held-out fillers)")
            ax.axvline(SWEEP.index(32), color=C_CROWD, lw=0.8, ls=":")
            ax.axvline(0, color=C_SOLO, lw=0.8, ls=":")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("fraction answering Yes / owl (crowded)")
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].legend(fontsize=7)
        axes[1].set_ylabel("mean log-odds [nats] (95% bootstrap CI)")
        axes[1].legend(fontsize=7, loc="center", bbox_to_anchor=(0.55, 0.27), ncol=2)
        axes[0].set_title("Trained only on B=1 vs B=32", fontsize=9)
        axes[1].set_title("Log-odds vs batch size", fontsize=9)
        if pert:
            ax = axes[2]
            rel = [np.mean([r["B"][str(B)]["rel_l2_vs_B1"] for r in pert["rows"]]) for B in SWEEP]
            same32 = [np.mean([r["B"][str(B)]["bit_identical_to_B32"] for r in pert["rows"]]) for B in SWEEP]
            ax.bar(xs, rel, color="#6c757d", alpha=0.8, label="‖h₃₄(B) − h₃₄(1)‖ / ‖h₃₄(1)‖")
            ax.set_ylim(0, max(rel) * 1.45)
            ax.set_ylabel("relative L2 perturbation of layer-34 input vs B = 1")
            ax2 = ax.twinx()
            ax2.plot(xs, same32, color=C_CROWD, marker="D", ms=4,
                     label="fraction bit-identical to B = 32")
            ax2.set_ylim(-0.03, 1.03)
            ax2.set_ylabel("fraction of prompts with h₃₄ bit-identical to B = 32", color=C_CROWD)
            ax.set_xticks(xs, [str(b) for b in SWEEP])
            ax.set_xlabel("batch size B")
            ax.set_title("Raw numerical footprint entering layer 34", fontsize=9)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", framealpha=0.9)
            summary["perturbation"] = {B: {"rel_l2_mean": float(r), "frac_bit_identical_to_B32": float(s),
                                           "frac_bit_identical_to_B1": float(np.mean(
                                               [x["B"][str(B)]["bit_identical_to_B1"] for x in pert["rows"]]))}
                                       for B, r, s in zip(SWEEP, rel, same32)}
        fig.tight_layout()
        save(fig, "fig2_batch_size_sweep.png")
        plt.close(fig)

    # ---------------- fig3: mechanistic patching ----------------
    if patch:
        rows = patch["rows"]
        mech = {}
        fig = plt.figure(figsize=(15, 8.5))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
        # (a) factorial
        ax = fig.add_subplot(gs[0, 0])
        labels = [("solo_input_solo_geom", "input B=1\ntail B=1"), ("crowd_input_solo_geom", "input B=32\ntail B=1"),
                  ("solo_input_crowd_geom", "input B=1\ntail B=32"), ("crowd_input_crowd_geom", "input B=32\ntail B=32")]
        mech["factorial"] = {}
        for k_task, task in enumerate(TASKS):
            rs = [r for r in rows if r["task"] == task]
            vals = [bootstrap_mean([r["factorial"][k] for r in rs], 2000) for k, _ in labels]
            mech["factorial"][task] = {k: v for (k, _), v in zip(labels, vals)}
            ax.bar(np.arange(4) + (k_task - 0.5) * 0.38, [v["mean"] for v in vals], 0.38,
                   yerr=[[v["mean"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["mean"] for v in vals]],
                   color=[C_SOLO, "#9b5de5", "#f4a261", C_CROWD], alpha=0.55 + 0.4 * k_task,
                   label=f"{POS[task]}−{NEG[task]}", edgecolor="k", lw=0.4)
        ax.set_xticks(np.arange(4), [l for _, l in labels], fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("mean log-odds [nats]")
        ax.set_title("(a) Where is the footprint? input × tail geometry", fontsize=9)
        ax.legend(fontsize=7)
        # (b) token-position patching (solo geometry, denoising)
        geoms = list(rows[0]["positions"].keys())
        ax = fig.add_subplot(gs[0, 1:])
        names = ([f"{n} [{i}]" for i, n in enumerate(patch["prefix_names"])] + ["prompt body (all)"] +
                 [f"{n} [L{q - 9}]" for q, n in enumerate(patch["suffix_names"])])
        assert len(set(names)) == len(names)
        width = 0.8 / max(1, len(geoms) * len(TASKS))
        mech["positions"] = {}
        for gi, geom in enumerate(geoms):
            for ti, task in enumerate(TASKS):
                rs = [r for r in rows if r["task"] == task]
                fr = []
                for r in rs:
                    P = r["positions"][geom]
                    tot = P["full"] - P["base"]
                    L = r["L"]
                    single = {int(k): v for k, v in P["single"].items()}
                    vals = [single[p] for p in range(3)] + [P["body_all"]] + [single[L - 9 + q] for q in range(9)]
                    fr.append([(v - P["base"]) / tot if abs(tot) > 1e-6 else np.nan for v in vals])
                fr = np.array(fr)
                m = np.nanmean(fr, 0)
                mech["positions"][f"{task}_{geom}_geom"] = dict(zip(names, map(float, m)))
                off = (gi * len(TASKS) + ti - (len(geoms) * len(TASKS) - 1) / 2) * width
                ax.bar(np.arange(len(names)) + off, m, width,
                       label=f"{POS[task]}−{NEG[task]}, tail at B={'1' if geom == 'solo' else '32'}",
                       color=[C_SOLO, C_CROWD][gi], alpha=0.5 + 0.4 * ti, edgecolor="k", lw=0.3)
        ax.set_xticks(np.arange(len(names)), names, rotation=35, fontsize=7, ha="right")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(1, color="k", lw=0.6, ls=":")
        ax.set_ylabel("fraction of Δ log-odds restored")
        ax.set_title("(b) Patch the B=32 layer-34 input into the B=1 run at one position (or the whole body)", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        # (c) heads heatmap
        ax = fig.add_subplot(gs[1, :2])
        geom = geoms[0]
        mats, ylabels = [], []
        mech["heads"] = {}
        for task in TASKS:
            rs = [r for r in rows if r["task"] == task]
            for key in sorted(rs[0]["heads"][geom].keys()):
                fr = []
                for r in rs:
                    P = r["positions"][geom]
                    tot = P["full"] - P["base"]
                    fr.append([(v - P["base"]) / tot if abs(tot) > 1e-6 else np.nan
                               for v in r["heads"][geom][key]])
                m = np.nanmean(np.array(fr), 0)
                mats.append(m)
                ylabels.append(f"{task[:3]} L{key.replace('_', ' ')}")
                mech["heads"][f"{task}_L{key}_{geom}_geom"] = list(map(float, m))
        M = np.array(mats)
        vmax = max(0.05, float(np.nanmax(np.abs(M))))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
        ax.set_xticks(range(32), [str(h) for h in range(32)], fontsize=6)
        ax.set_xlabel("attention head")
        ax.set_title(f"(c) Head patching (tail at B={'1' if geom == 'solo' else '32'}): "
                     "fraction of Δ restored by one head's output from the B=32 run", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02)
        # (d) components
        ax = fig.add_subplot(gs[1, 2])
        comps = sorted(rows[0]["components"][geom].keys())
        mech["components"] = {}
        for ti, task in enumerate(TASKS):
            rs = [r for r in rows if r["task"] == task]
            m = []
            for c in comps:
                fr = []
                for r in rs:
                    P = r["positions"][geom]
                    tot = P["full"] - P["base"]
                    fr.append((r["components"][geom][c] - P["base"]) / tot if abs(tot) > 1e-6 else np.nan)
                m.append(float(np.nanmean(fr)))
            mech["components"][f"{task}_{geom}_geom"] = dict(zip(comps, m))
            ax.bar(np.arange(len(comps)) + (ti - 0.5) * 0.38, m, 0.38, label=f"{POS[task]}−{NEG[task]}",
                   color=C_TRAIN, alpha=0.5 + 0.4 * ti, edgecolor="k", lw=0.3)
        ax.set_xticks(np.arange(len(comps)), comps, fontsize=8)
        ax.axhline(1, color="k", lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("(d) Component patching", fontsize=9)
        ax.set_ylabel("fraction of Δ restored")
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, "fig3_mechanistic_patching.png", 170)
        plt.close(fig)
        summary["mechanism"] = mech

    # ---------------- fig0: Gate 1 numerical audit ----------------
    if gate1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for e in gate1["prompts"]:
            axes[0].plot([x["frac_elements_differ"] for x in e["layers_solo_vs_batchA"]], color=C_CROWD,
                         alpha=0.5, lw=1)
            axes[0].plot([x["frac_elements_differ"] for x in e["layers_batchA_vs_batchB"]], color="k",
                         alpha=0.5, lw=1)
        axes[0].plot([], [], color=C_CROWD, label="B=1 vs B=32 (filler set A)")
        axes[0].plot([], [], color="k", label="B=32 set A vs B=32 set B (identically 0)")
        axes[0].set_xlabel("decoder layer (output)")
        axes[0].set_ylabel("fraction of target-row elements that differ")
        axes[0].set_ylim(-0.02, 1.0)
        axes[0].set_title("Gate 1: batch-size footprint per layer (8 prompts)", fontsize=9)
        axes[0].legend(fontsize=7, loc="lower right")
        axes[0].grid(alpha=0.25)
        ops = gate1["op_isolation_prompt0"]
        keys = [k for k in ops if k.startswith("layers.34.") or k in ("lm_head", "norm")]
        vals = [ops[k]["frac_elements_differ"] for k in keys]
        cols = [C_CROWD if v > 0 else (C_TRAIN if ops[k]["kernel_changed"] else C_BASE) for k, v in zip(keys, vals)]
        axes[1].barh(range(len(keys)), vals, color=cols)
        for yi, k in enumerate(keys):
            if ops[k]["kernel_changed"] and ops[k]["frac_elements_differ"] == 0:
                axes[1].plot(0.006, yi, marker="o", color=C_TRAIN)
            if any("splitK" in x for x in ops[k]["kernels_solo"]):
                axes[1].text(ops[k]["frac_elements_differ"] + 0.01, yi, "split-K at M=L", va="center",
                             fontsize=7)
        axes[1].set_xlim(0, 0.62)
        axes[1].set_yticks(range(len(keys)), [k.replace("layers.34.", "") for k in keys], fontsize=7)
        axes[1].set_xlabel("fraction of output elements that change\n(identical input row; M = L vs M = 32 L)")
        axes[1].set_title("Op isolation, layer 34 (prompt 0): red bar = bits change;\n"
                          "green dot = kernel switches but bits identical; none = same kernel", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{topdir}/fig0_gate1_audit.png", dpi=170)
        plt.close(fig)

    # ---------------- training curve ----------------
    log_path = f"{root}/train_log_{tag}.jsonl"
    if os.path.exists(log_path):
        log = [json.loads(l) for l in open(log_path)]
        ev = [e for e in log if "val" in e]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for task, ls in (("detection", "-"), ("animal", "--")):
            axes[0].plot([e["step"] for e in ev], [e["val"][task]["accuracy"] for e in ev], ls=ls,
                         color=C_TRAIN, marker="o", ms=3, label=f"{task}: accuracy")
            axes[0].plot([e["step"] for e in ev], [e["val"][task]["both"] / e["val"][task]["n"] for e in ev],
                         ls=ls, color=C_CROWD, marker="s", ms=3, label=f"{task}: both answers right")
            axes[1].plot([e["step"] for e in ev], [e["val"][task]["mean_delta"] for e in ev], ls=ls,
                         color=C_TRAIN, marker="o", ms=3, label=f"{task}: mean Δ log-odds")
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].axhline(0.5, color="k", lw=0.6, ls=":")
        axes[0].set_xlabel("update")
        axes[0].set_title("Validation (64 prompt pairs per task)", fontsize=9)
        axes[0].legend(fontsize=7)
        axes[1].set_xlabel("update")
        axes[1].set_ylabel("nats")
        axes[1].legend(fontsize=7)
        tr = [e for e in log if "ce" in e]
        ax2 = axes[1].twinx()
        ax2.plot([e["step"] for e in tr], [e["ce"] for e in tr], color=C_BASE, lw=0.6, alpha=0.6)
        ax2.set_ylabel("train CE (grey)")
        fig.tight_layout()
        save(fig, "fig4_training_curve.png", 170)
        plt.close(fig)
        summary["training_log_last_eval"] = ev[-1] if ev else None

    if kmap:
        km = {}
        for name, v in kmap["shapes"].items():
            seqs = {int(m): tuple(k) for m, k in v["kernels_by_M"].items()}
            changes, prev = [], None
            for m in sorted(seqs):
                if seqs[m] != prev:
                    changes.append({"from_M": m, "kernels": list(seqs[m])})
                    prev = seqs[m]
            km[name] = {"N": v["N"], "K": v["K"], "regimes": changes,
                        "split_k_M": sorted(m for m, k in seqs.items() if any("splitK" in x for x in k))}
        summary["kernel_map"] = km
    with open(f"{root}/summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    return {k: v for k, v in summary.items() if k != "kernel_map"}
