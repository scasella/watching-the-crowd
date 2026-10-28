"""Phase 6 statistics and figures (fig5: M = B x L confound, fig6: defenses). Runs on Modal.

Library module (no CLI): imported by 06_controls_and_defenses.py (function analyze6), which writes
phase6_controls_defenses.json, fig5 and fig6.
"""

import json
import os

TASKS = (("detection", "Yes", "No"), ("animal", "owl", "cat"))
C = {"base": "#8c8c8c", "main": "#7fb7b0", "scaled": "#2a9d8f", "solo": "#3b6fb6", "crowd": "#d1495b"}


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def boot(vals, n=10_000, seed=42):
    import numpy as np

    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def paired(rows, task, a="B1", b="B32", key=None):
    """rows: [{task, B1: answer, B32: answer}] (or nested under key). Detector metrics."""
    rs = [r for r in rows if r["task"] == task]
    if not rs:
        return {"n": 0, "correct": 0, "answers": 0, "accuracy": 0.0, "solo_correct": 0, "crowd_correct": 0,
                "both": 0, "delta": {"mean": 0.0, "ci95": [0.0, 0.0]}, "frac_delta_positive": 0.0}
    get = (lambda r, s: r[key][s]) if key else (lambda r, s: r[s])
    d = [get(r, b)["log_odds"] - get(r, a)["log_odds"] for r in rs]
    k1 = sum(get(r, a)["top_is_neg"] for r in rs)
    k32 = sum(get(r, b)["top_is_pos"] for r in rs)
    return {"n": len(rs), "correct": k1 + k32, "answers": 2 * len(rs), "accuracy": (k1 + k32) / (2 * len(rs)),
            "solo_correct": k1, "crowd_correct": k32,
            "both": sum(get(r, a)["top_is_neg"] and get(r, b)["top_is_pos"] for r in rs),
            "delta": boot(d), "frac_delta_positive": sum(x > 0 for x in d) / len(d)}


def run(root, suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from safetensors.torch import load_file

    figdir = os.path.join(root, "figures")
    os.makedirs(figdir, exist_ok=True)
    prep = _load(f"{root}/phase6_prep.json")
    fb = _load(f"{root}/phase6_factor_bits{suffix}.json")
    sw = _load(f"{root}/phase6_sweep{suffix}.json")
    dec = _load(f"{root}/phase6_decode{suffix}.json")
    dfn = _load(f"{root}/phase6_defenses{suffix}.json")
    flags = {c: _load(f"{root}/phase6_flags_{c}{suffix}.json") for c in ("default", "no_rpr", "det_4096", "det_16")}
    out = {}

    # ---------------- exact kernel map ----------------
    if prep:
        km = prep["kernel_map_exact"]
        out["split_k_ranges"] = {k: v["split_k_ranges"] for k, v in km.items()}
        out["kernel_regime_counts"] = {k: len(v["regimes"]) for k, v in km.items()}
        out["pos2_bank"] = {c: {e["label"]: {"M": e["M"], "n": e["n"], "splits": e["splits"]} for e in d.values()}
                            for c, d in prep["pos2_bank"].items()}

    # ---------------- 1A: factorizations ----------------
    if fb:
        fa = {}
        for M, r in fb["M"].items():
            keys = [f"{b}x{l}" for b, l in r["factorizations"]]
            row = {}
            for k in keys:
                ents = [e["vs_long_solo"][k] for e in r["per_prompt"]]
                ops = sorted({(e["first_differing_op"] or {}).get("op", "none") for e in ents})
                preds = [e["predictions"][k] for e in r["per_prompt"] if e["predictions"][k]["prompt_fits"]]
                row[k] = {"pos012_identical_to_long_solo": sum(e["h34_pos012"]["bit_identical"] for e in ents),
                          "pos2_identical_to_long_solo": sum(e["h34_pos2"]["bit_identical"] for e in ents),
                          "n": len(ents), "first_differing_ops": ops,
                          "pos012_linf_max": max(e["h34_pos012"]["linf"] for e in ents),
                          "pos012_frac_differ_mean": float(np.mean([e["h34_pos012"]["frac_differ"] for e in ents])),
                          "pos2_hashes": sorted({e["h34_pos2_hash"] for e in ents}),
                          "prompts_fit": len(preds),
                          "scaled_says_crowded": sum(p["scaled"]["top_is_pos"] for p in preds),
                          "kernels": r["kernels"].get(k)}
            fa[M] = row
        out["factorizations"] = fa

    # ---------------- 1B: solo length sweep ----------------
    if sw:
        rows = sw["rows"]
        Ls = sorted({int(L) for r in rows for L in r["solo"]})
        nat_key = lambda r: str(r["L_natural"])  # noqa: E731
        sweep = {}
        for task, pos, neg in TASKS:
            sweep[task] = {}
            for L in Ls:
                rs = [r for r in rows if r["task"] == task and str(L) in r["solo"] and int(L) > r["L_natural"]]
                ent = {"n": len(rs)}
                for tag in ("base", "main", "scaled"):
                    if not rs:
                        continue
                    lo = [r["solo"][str(L)][tag]["log_odds"] for r in rs]
                    d = [r["solo"][str(L)][tag]["log_odds"] - r["solo"][nat_key(r)][tag]["log_odds"] for r in rs]
                    ent[tag] = {"says_crowded": sum(r["solo"][str(L)][tag]["top_is_pos"] for r in rs),
                                "says_solo": sum(r["solo"][str(L)][tag]["top_is_neg"] for r in rs),
                                "mean_log_odds": float(np.mean(lo)), "delta_vs_natural": boot(d, 2000)}
                if rs:
                    ent["trained_minus_base_delta"] = {
                        t: ent[t]["delta_vs_natural"]["mean"] - ent["base"]["delta_vs_natural"]["mean"]
                        for t in ("main", "scaled")}
                    p2 = [r["solo"][str(L)] for r in rs]
                    ent["pos2_unique_hashes"] = len({e["pos2_hash"] for e in p2})
                    ent["pos2_bank"] = sorted({f"{e['pos2_bank']['cond']}:{e['pos2_bank']['label']}" for e in p2})
                    ent["pos2_frac_differ_vs_32x32"] = float(np.mean([e["pos2_frac_differ_vs_32x32"] for e in p2]))
                    ent["pos012_identical_to_32x32"] = sum(e["pos012_identical_to_32x32"] for e in p2)
                sweep[task][L] = ent
            nat = [r for r in rows if r["task"] == task]
            sweep[task]["natural"] = {t: {"says_crowded": sum(r["solo"][nat_key(r)][t]["top_is_pos"] for r in nat),
                                          "says_solo": sum(r["solo"][nat_key(r)][t]["top_is_neg"] for r in nat),
                                          "n": len(nat)} for t in ("base", "main", "scaled")}
        out["solo_sweep"] = sweep
        f3 = {}
        keys = sorted({k for r in rows for k in r["f3072"]}, key=lambda k: -int(k.split("x")[0]))
        for task, pos, neg in TASKS:
            f3[task] = {}
            for k in keys:
                rs = [r for r in rows if r["task"] == task and k in r["f3072"]]
                f3[task][k] = {t: {"says_crowded": sum(r["f3072"][k][t]["top_is_pos"] for r in rs),
                                   "mean_log_odds": float(np.mean([r["f3072"][k][t]["log_odds"] for r in rs]))}
                               for t in ("base", "main", "scaled")}
                f3[task][k]["n"] = len(rs)
                f3[task][k]["pos2_bank"] = sorted({f"{r['f3072'][k]['pos2_bank']['cond']}:{r['f3072'][k]['pos2_bank']['label']}" for r in rs})
        same = 0
        for r in rows:
            hs = {r["f3072"][k]["pos012_hash"] for k in r["f3072"]}
            same += len(hs) == 1
        out["f3072"] = {"per_task": f3, "prompts_with_identical_pos012_across_factorizations": same,
                        "n": len(rows)}

    # ---------------- 1C: decode ----------------
    if dec:
        dd = {}
        for L, per in dec["L"].items():
            ent = {}
            for k in per[0]["vs_P1_D1"]:
                ent[k] = {m: float(np.mean([e["vs_P1_D1"][k][m] for e in per])) for m in ("linf", "rel_l2", "frac_differ")}
                ent[k]["bit_identical"] = sum(e["vs_P1_D1"][k]["bit_identical"] for e in per)
            ent["first_differing_op"] = sorted({(e["decode_first_differing_op"] or {}).get("op", "none") for e in per})
            ent["copies_equal_real_companions"] = sum(e["copies_equal_real_companions"] for e in per)
            ent["prefill_kv34_carryover_frac_differ"] = float(np.mean([e["prefill_kv_layer34_carryover"]["frac_differ"] for e in per]))
            p0 = per[0]
            ent["decode_kernels"] = p0.get("decode_kernels")
            ent["sdpa_isolation_layer0"] = p0.get("sdpa_isolation_layer0")
            ent["sdpa_kernels"] = p0.get("sdpa_kernels")
            ent["gemm_isolation_layer0"] = p0.get("gemm_isolation_layer0")
            ent["n"] = len(per)
            dd[L] = ent
        out["decode"] = dd

    # ---------------- 2A: flags ----------------
    if all(flags.values()):
        ref_ord = load_file(f"{root}/phase6_flags_ordinary_default.safetensors")
        ref_h = load_file(f"{root}/phase6_flags_h34_default.safetensors")
        fl = {}
        for cfg, f in flags.items():
            ord_ = load_file(f"{root}/phase6_flags_ordinary_{cfg}.safetensors")
            hh = load_file(f"{root}/phase6_flags_h34_{cfg}.safetensors")
            util = {}
            for B in (1, 32):
                ks = [k for k in ref_ord if k.endswith(f"_B{B}") and k in ord_]
                kl = [float((ref_ord[k].exp() * (ref_ord[k] - ord_[k])).sum()) for k in ks]
                agree = [int(ref_ord[k].argmax() == ord_[k].argmax()) for k in ks]
                util[f"B{B}"] = {"kl_last_position_vs_default": float(np.mean(kl)), "top1_agree": float(np.mean(agree))}
            h_same = {B: sum(bool((ref_h[k] == hh[k]).all()) for k in ref_h if k.endswith(f"_B{B}") and k in hh)
                      for B in (1, 32)}
            fl[cfg] = {"gate_bit_identical": sum(g["bit_identical"] for g in f["gate"]), "gate_n": len(f["gate"]),
                       "gate_frac_differ_mean": float(np.mean([g["frac_differ"] for g in f["gate"]])),
                       "gate_rel_l2_mean": float(np.mean([g["rel_l2"] for g in f["gate"]])),
                       "gate_linf_max": max(g["linf"] for g in f["gate"]),
                       "kernels": f["kernels"], "deterministic": f.get("deterministic_algorithms"),
                       "detector": {t: paired(f["eval"], t) for t, _, _ in TASKS},
                       "ordinary_utility": util,
                       "h34_identical_to_default_flags": {f"B{B}": n for B, n in h_same.items()},
                       "cublas_workspace_config": f["env"].get("cublas_workspace_config")}
        out["flags"] = fl
        # yardstick: how much the undefended model's own next-token distribution moves B=1 -> B=32
        kl_b = []
        agree_b = []
        for k in ref_ord:
            if k.endswith("_B1") and k.replace("_B1", "_B32") in ref_ord:
                a, b = ref_ord[k], ref_ord[k.replace("_B1", "_B32")]
                kl_b.append(float((a.exp() * (a - b)).sum()))
                agree_b.append(int(a.argmax() == b.argmax()))
        out["natural_batch_kl_last_position"] = {"kl_B1_vs_B32": float(np.mean(kl_b)) if kl_b else None,
                                                 "top1_agree": float(np.mean(agree_b)) if agree_b else None,
                                                 "model": "scaled tail, default flags"}

    # ---------------- 2B / 2C ----------------
    if dfn:
        pin = {}
        for lv in ("none", "pin2", "pin012", "pin_template", "pin012_fixedprefix"):
            pin[lv] = {t: paired(dfn["pin_l34"], t, key=lv) for t, _, _ in TASKS}
        for lv in ("kv2_solo_ref", "kv012_solo_ref", "kv2_fixed_ref", "kv012_fixed_ref"):
            pin[lv] = {t: paired(dfn["pin_kv"], t, key=lv) for t, _, _ in TASKS}
        out["pinning"] = pin
        out["pinning_kv_utility"] = dfn.get("pin_kv_utility")
        jam = []
        for j in dfn["jamming"]:
            jam.append({"scope": j["scope"], "k": j["k"], "utility": j["utility"],
                        "pos2_bank_match": j["pos2_bank_match"],
                        "detector": {t: paired(j["rows"], t) for t, _, _ in TASKS}})
        out["jamming"] = jam

    # ================= fig5 =================
    if fb or sw:
        fig = plt.figure(figsize=(19, 5.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.25])
        ax = fig.add_subplot(gs[0, 0])
        if fb:
            Ms = sorted(out["factorizations"], key=int)
            mat, text, ylabels = [], [], []
            for i in range(6):
                rowv, rowt = [], []
                for M in Ms:
                    k = list(out["factorizations"][M])[i]
                    e = out["factorizations"][M][k]
                    rowv.append(e["pos012_identical_to_long_solo"] / e["n"])
                    op = [o for o in e["first_differing_ops"] if o != "none"]
                    rowt.append(f"[{k.replace('x', ', ')}]\n{e['pos012_identical_to_long_solo']}/{e['n']}"
                                + (f"\n≠ at {op[0].split('.', 1)[1]} L{op[0].split('.')[0]}" if op else ""))
                mat.append(rowv)
                text.append(rowt)
                ylabels.append(f"B = {list(out['factorizations'][Ms[0]])[i].split('x')[0]}")
            im = ax.imshow(np.array(mat), cmap="Greens", vmin=0, vmax=2.2, aspect="auto")
            for i, rowt in enumerate(text):
                for j, t in enumerate(rowt):
                    ax.text(j, i, t.replace("\n", " · ") if "≠" not in t else t, ha="center", va="center",
                            fontsize=8.5)
            ax.set_xticks(range(len(Ms)), [f"M = {m}" for m in Ms])
            ax.set_yticks(range(6), ylabels)
            ax.set_title("A. Layer-34 state at positions 0–2: [B, L] vs [1, M]\n"
                         "(cell: prompts bit-identical to the long solo run; first differing op if not)", fontsize=9)
        if sw:
            sk = out.get("split_k_ranges", {})

            def shade(ax):
                for name, color, yb in (("k_proj/v_proj", "#f4a261", (0.0, 0.5)), ("down_proj", "#9b5de5", (0.5, 1.0)),
                                        ("q_proj/o_proj", "#2a9d8f", (0.0, 0.06))):
                    for a, b in sk.get(name, []):
                        if b >= 30:
                            ax.axvspan(max(a, 30), b + 1, ymin=yb[0], ymax=yb[1], color=color, alpha=0.16, lw=0)
                for name, color in (("k/v_proj split-K (lower band)", "#f4a261"),
                                    ("down_proj split-K (upper band)", "#9b5de5"),
                                    ("q/o_proj split-K (thin strip)", "#2a9d8f")):
                    ax.fill_between([], [], color=color, alpha=0.3, label=name)
                ax.set_xscale("log")
                ax.set_xlabel("solo prompt length L = M (B = 1); first point = natural length")
                ax.grid(alpha=0.25)

            axb = fig.add_subplot(gs[0, 1])
            axc = fig.add_subplot(gs[0, 2])
            shade(axb)
            shade(axc)
            for task, pos, neg in TASKS:
                st = out["solo_sweep"][task]
                if st["natural"]["base"]["n"] == 0:
                    continue
                Ls = sorted([L for L in st if L != "natural" and st[L]["n"] > 0], key=float)
                xs = [42.0] + [float(L) for L in Ls]
                ls = "-" if task == "detection" else "--"
                mk = "o" if task == "detection" else "s"
                for tag in ("base", "main", "scaled"):
                    nat = st["natural"][tag]
                    ys = [nat["says_crowded"] / nat["n"]] + [st[L][tag]["says_crowded"] / st[L]["n"] for L in Ls]
                    axb.plot(xs, ys, ls=ls, marker=mk, ms=2.5, color=C[tag], label=f"{tag}: says {pos}")
                    dl = [0.0] + [st[L][tag]["delta_vs_natural"]["mean"] for L in Ls]
                    axc.plot(xs, dl, ls=ls, marker=mk, ms=2.5, color=C[tag], label=f"{tag}: {pos}−{neg}")
            axb.set_ylabel('fraction answering Yes / owl ("crowded")')
            axb.set_ylim(-0.03, 1.03)
            axb.set_title("B. Alone on the GPU, lengthened with neutral text:\nfraction of answers that say \"crowded\"", fontsize=9)
            axb.legend(fontsize=6, ncol=2, loc="upper left")
            axc.axhline(0, color="k", lw=0.6)
            axc.set_ylabel("mean Δ log-odds vs. the same prompt at natural length [nats]")
            axc.set_title("C. Shift in the answer's log-odds as the solo prompt grows", fontsize=9)
            axc.legend(fontsize=6, ncol=2, loc="upper left")
        fig.tight_layout()
        fig.savefig(f"{figdir}/fig5_length_vs_batch_confound{suffix}.png", dpi=170)
        plt.close(fig)

    # ================= fig6 =================
    if dfn or all(flags.values()):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6.0), gridspec_kw={"width_ratios": [1.9, 1]})
        ax = axes[0]
        bars = []
        if all(flags.values()):
            for cfg, lab in (("default", "baseline\n(default flags)"), ("no_rpr", "bf16 reduced-\nprecision\nreduction off"),
                             ("det_4096", "deterministic\n+ ws :4096:8"), ("det_16", "deterministic\n+ ws :16:8")):
                bars.append((lab, out["flags"][cfg]["detector"]))
        if dfn:
            for lv, lab in (("pin2", "pin pos 2\n(L34 input)"), ("pin012", "pin pos 0–2\n(L34 input)"),
                            ("pin_template", "pin all template\ntokens (L34)"),
                            ("pin012_fixedprefix", "pin pos 0–2 to\nfixed prefix (L34)"),
                            ("kv012_solo_ref", "prefix-cache K/V\npos 0–2 (B=1 ref)"),
                            ("kv012_fixed_ref", "prefix-cache K/V\npos 0–2 (fixed)")):
                bars.append((lab, out["pinning"][lv]))
        x = np.arange(len(bars))
        for ti, (task, pos, neg) in enumerate(TASKS):
            m = [b[1][task]["delta"]["mean"] for b in bars]
            lo = [b[1][task]["delta"]["mean"] - b[1][task]["delta"]["ci95"][0] for b in bars]
            hi = [b[1][task]["delta"]["ci95"][1] - b[1][task]["delta"]["mean"] for b in bars]
            ax.bar(x + (ti - 0.5) * 0.38, m, 0.38, yerr=[lo, hi], color=C["scaled"], alpha=0.5 + 0.4 * ti,
                   edgecolor="k", lw=0.3, label=f"{pos} − {neg}")
            for xi, b in zip(x, bars):
                acc = b[1][task]["accuracy"]
                ax.text(xi + (ti - 0.5) * 0.38, -2.2, f"{100 * acc:.0f}%", ha="center", va="top", fontsize=6.5,
                        rotation=90)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [b[0] for b in bars], fontsize=6.5)
        ax.set_ylabel("Δ log-odds, B = 32 − B = 1 [nats] (95% CI)\nnumbers below bars: top-1 accuracy")
        ax.set_title("A. How much of the separation survives each defense (scaled checkpoint, 64+64 test prompts)", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1]
        if dfn:
            for scope, mk, col in (("template", "o", C["solo"]), ("all", "s", C["crowd"])):
                pts = [j for j in out["jamming"] if j["scope"] == scope]
                if not pts:
                    continue
                for task, pos, neg in TASKS:
                    xs = [max(j["utility"]["B1"]["kl_per_position"], 1e-7) for j in pts]
                    ys = [j["detector"][task]["delta"]["mean"] for j in pts]
                    ax.plot(xs, ys, marker=mk, color=col, ls="-" if task == "detection" else "--",
                            label=f"{scope} jitter: {pos}−{neg}")
                    for xx, yy, j in zip(xs, ys, pts):
                        ax.annotate(f"k={j['k']}", (xx, yy), fontsize=6, xytext=(3, 3), textcoords="offset points")
            kvu = (out.get("pinning_kv_utility") or {}).get("kv012_fixed_ref")
            if kvu:
                for task, pos, neg in TASKS:
                    ax.scatter([kvu["B1"]["kl_per_position"]], [out["pinning"]["kv012_fixed_ref"][task]["delta"]["mean"]],
                               marker="*", s=140, color="#e9c46a", edgecolor="k", zorder=5,
                               label="prefix-cache K/V pos 0–2 (fixed)" if task == "detection" else None)
            nat = (out.get("natural_batch_kl_last_position") or {}).get("kl_B1_vs_B32")
            if nat:
                ax.axvline(nat, color="#6c757d", ls=":", lw=1)
                ax.text(nat, 10, " undefended model's own\n B=1 vs B=32 KL\n (last position)", fontsize=7, color="#6c757d")
            none = [j for j in out["jamming"] if j["scope"] == "none"]
            if none:
                for task, pos, neg in TASKS:
                    ax.axhline(none[0]["detector"][task]["delta"]["mean"], color="k", lw=0.6,
                               ls="-" if task == "detection" else "--")
            ax.set_xscale("log")
            ax.set_xlabel("utility cost: KL(noiseless ‖ jammed) per position, 64 ordinary prompts, B = 1 [nats]")
            ax.set_ylabel("Δ log-odds, B = 32 − B = 1 [nats]")
            ax.axhline(0, color="k", lw=0.4, ls=":")
            ax.set_title("B. Stochastic k-ULP jitter: suppression vs. cost (black lines: no jitter)", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{figdir}/fig6_defenses{suffix}.png", dpi=170)
        plt.close(fig)

    with open(f"{root}/phase6_controls_defenses{suffix}.json", "w") as f:
        json.dump(out, f, indent=1)
    return {k: (v if k in ("split_k_ranges",) else "…") for k, v in out.items()}
