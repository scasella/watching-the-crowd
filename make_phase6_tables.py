"""Markdown tables for REPORT.md Sections 7-8 from results_watching_crowd/phase6_controls_defenses.json (stdlib only).

    python3 make_phase6_tables.py {factor | f3072 | sweep | decode | flags | pin | jam}
"""

import json
import os
import sys

S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results_watching_crowd", "phase6_controls_defenses.json")))
TASKS = (("detection", "Yes", "No"), ("animal", "owl", "cat"))


def ci(d):
    return f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"


def factor_table():
    out = ["| M | [B, L] factorizations | Positions 0–2 bit-identical to [1, M] | First differing op (36 layers) | FlashAttention kernel | split-K launches | Scaled detector says \"crowded\" (8 prompts) |",
           "|---|---|---|---|---|---|---|"]
    for M, row in S["factorizations"].items():
        keys = list(row)
        ident = [f"{row[k]['pos012_identical_to_long_solo']}/{row[k]['n']}" for k in keys]
        ops = sorted({o for k in keys for o in row[k]["first_differing_ops"]})
        flash = sorted({n.split("<")[0].split("::")[-1] + "<" + n.split("<")[2].split(",")[0:4].__str__()
                        for k in keys for n in (row[k]["kernels"] or {}).get("flash", {})})
        sk = sorted({(row[k]["kernels"] or {}).get("splitKreduce") for k in keys})
        shape = {k: "[" + k.replace("x", ", ") + "]" for k in keys}
        pred = ", ".join(f"{shape[k]}: {row[k]['scaled_says_crowded']}/{row[k]['prompts_fit']}" if row[k]["prompts_fit"]
                         else f"{shape[k]}: no prompt fits" for k in keys)
        out.append(f"| {M} | {', '.join('[' + k.replace('x', ', ') + ']' for k in keys)} | {' · '.join(ident)} | "
                   f"{'none' if ops == ['none'] else ', '.join(ops)} | same `flash_fwd_kernel` (×36) in all | {', '.join(map(str, sk))} | {pred} |")
    return "\n".join(out)


def f3072_table():
    f = S["f3072"]["per_task"]
    keys = list(f["detection"])
    out = ["| Model | Task | " + " | ".join("[" + k.replace("x", ", ") + "]" for k in keys) + " |",
           "|---|---|" + "---|" * len(keys)]
    for tag in ("base", "main", "scaled"):
        for task, pos, neg in TASKS:
            out.append(f"| {tag} | says {pos} (of 64) | " + " | ".join(str(f[task][k][tag]["says_crowded"]) for k in keys) + " |")
    out.append(f"| | position-2 pattern | " + " | ".join(", ".join(x.split(":")[1] + (" (crowded)" if x.startswith("crowd:") else " (solo)" if x.startswith("solo:") else "")
                                                               for x in f["detection"][k]["pos2_bank"]) for k in keys) + " |")
    return "\n".join(out)


def sweep_table(Ls):
    st = S["solo_sweep"]
    out = ["| Solo length L (= M) | Position-2 pattern | Scaled: Yes / owl (of 64) | Spec: Yes / owl | Base: Yes / owl | Scaled Δ vs natural, det / animal [nats] | Base Δ vs natural |",
           "|---|---|---|---|---|---|---|"]
    nat = {t: st[t]["natural"] for t, _, _ in TASKS}
    out.append(f"| natural (37–61) | solo (known) | {nat['detection']['scaled']['says_crowded']} / {nat['animal']['scaled']['says_crowded']} | "
               f"{nat['detection']['main']['says_crowded']} / {nat['animal']['main']['says_crowded']} | "
               f"{nat['detection']['base']['says_crowded']} / {nat['animal']['base']['says_crowded']} | 0 / 0 | 0 / 0 |")
    for L in Ls:
        d, a = st["detection"][str(L)], st["animal"][str(L)]
        pat = ", ".join(x.replace("new:new", "**new**").replace("solo:", "solo ").replace("crowd:", "crowded ") for x in d["pos2_bank"])
        if d.get("pos012_identical_to_32x32"):
            pat += f" (≡ [32, 32] batch, {d['pos012_identical_to_32x32']}/64)"
        out.append(f"| {L} | {pat} | {d['scaled']['says_crowded']} / {a['scaled']['says_crowded']} | "
                   f"{d['main']['says_crowded']} / {a['main']['says_crowded']} | {d['base']['says_crowded']} / {a['base']['says_crowded']} | "
                   f"{d['scaled']['delta_vs_natural']['mean']:+.1f} / {a['scaled']['delta_vs_natural']['mean']:+.1f} | "
                   f"{d['base']['delta_vs_natural']['mean']:+.1f} / {a['base']['delta_vs_natural']['mean']:+.1f} |")
    return "\n".join(out)


def decode_table():
    dd = S["decode"]
    names = {"P1_D32copies": "prefill B=1, decode B=32 (copies of the B=1 cache)",
             "P1_D32real": "prefill B=1, decode B=32 (real companion caches)",
             "P32_D1": "prefill B=32, decode B=1 (carry-over only)",
             "P32_D32": "prefill B=32, decode B=32"}
    out = ["| vs. prefill B=1, decode B=1 | L = 48: elements differ / rel. L2 | L = 1024: elements differ / rel. L2 |", "|---|---|---|"]
    for k, lab in names.items():
        a, b = dd["48"][k], dd["1024"][k]
        out.append(f"| {lab} | {100 * a['frac_differ']:.1f} % / {a['rel_l2']:.4f} | {100 * b['frac_differ']:.1f} % / {b['rel_l2']:.4f} |")
    return "\n".join(out)


def flags_table():
    names = {"default": "Default (`allow_bf16_reduced_precision_reduction=True`, workspace default)",
             "no_rpr": "`allow_bf16_reduced_precision_reduction=False`",
             "det_4096": "`use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
             "det_16": "`use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:16:8`"}
    out = ["| Configuration | `splitKreduce` launches per forward, B=1 / B=32 | Layer-34 elements differing B=1 vs B=32 (8 prompts) | Detection: correct / Δ [95% CI] | Animal: correct / Δ [95% CI] | Ordinary-prompt KL vs default, B=1 / B=32 (last pos.) |",
           "|---|---|---|---|---|---|"]
    for cfg, lab in names.items():
        f = S["flags"][cfg]
        d, a = f["detector"]["detection"], f["detector"]["animal"]
        u = f["ordinary_utility"]
        out.append(f"| {lab} | {f['kernels']['B1']['splitKreduce']} / {f['kernels']['B32']['splitKreduce']} | "
                   f"{100 * f['gate_frac_differ_mean']:.1f} % ({f['gate_bit_identical']}/{f['gate_n']} identical) | "
                   f"{d['correct']}/128 / {ci(d['delta'])} | {a['correct']}/128 / {ci(a['delta'])} | "
                   f"{u['B1']['kl_last_position_vs_default']:.4f} / {u['B32']['kl_last_position_vs_default']:.4f} |")
    return "\n".join(out)


def pin_table():
    names = {"none": "No defense",
             "pin2": "Layer-34 input: position 2 ← its B=1 value (B=32 run only)",
             "pin012": "Layer-34 input: positions 0–2 ← B=1 values",
             "pin_template": "Layer-34 input: all 12 template positions ← B=1 values (body untouched)",
             "pin012_fixedprefix": "Layer-34 input: positions 0–2 ← one fixed prefix value (both runs)",
             "kv2_solo_ref": "Prefix-cache K/V, all 36 layers: position 2 ← B=1 values",
             "kv012_solo_ref": "Prefix-cache K/V, all 36 layers: positions 0–2 ← B=1 values",
             "kv2_fixed_ref": "Prefix-cache K/V, all 36 layers: position 2 ← one fixed cached prefix (both runs)",
             "kv012_fixed_ref": "Prefix-cache K/V, all 36 layers: positions 0–2 ← one fixed cached prefix (both runs)"}
    base = {t: S["pinning"]["none"][t]["delta"]["mean"] for t, _, _ in TASKS}
    out = ["| Pinning | Detection: correct / Δ [95% CI] / survives | Animal: correct / Δ [95% CI] / survives |", "|---|---|---|"]
    for lv, lab in names.items():
        p = S["pinning"][lv]
        cells = []
        for t, _, _ in TASKS:
            d = p[t]
            cells.append(f"{d['correct']}/128 / {ci(d['delta'])} / {100 * d['delta']['mean'] / base[t]:.0f} %")
        out.append(f"| {lab} | {cells[0]} | {cells[1]} |")
    return "\n".join(out)


def jam_table():
    out = ["| Jitter | Detection: correct / Δ [95% CI] | Animal: correct / Δ [95% CI] | Ordinary-prompt KL per position, B=1 / B=32 | Top-1 agreement, all positions (B=1) | Position-2 pattern in training bank |",
           "|---|---|---|---|---|---|"]
    for j in S["jamming"]:
        lab = "none" if j["scope"] == "none" else (f"positions 0–2 after every `down_proj` (layers 0–33), k = {j['k']} ULP" if j["scope"] == "template"
                                                   else f"all positions after `k/v/down_proj` (layers 0–33), k = {j['k']} ULP")
        d, a = j["detector"]["detection"], j["detector"]["animal"]
        u = j["utility"]
        m = j["pos2_bank_match"]
        out.append(f"| {lab} | {d['correct']}/128 / {ci(d['delta'])} | {a['correct']}/128 / {ci(a['delta'])} | "
                   f"{u['B1']['kl_per_position']:.4f} / {u['B32']['kl_per_position']:.4f} | {100 * u['B1']['top1_agree_all_positions']:.1f} % | "
                   f"B=1: {m['B1']['solo']} solo, {m['B1']['crowd']} crowded, {m['B1']['new']} new |")
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "factor":
        print(factor_table())
    elif which == "f3072":
        print(f3072_table())
    elif which == "sweep":
        print(sweep_table([64, 65, 96, 97, 128, 129, 192, 200, 256, 257, 260, 384, 512, 830, 831, 832, 960, 1024,
                           1255, 1256, 1300, 1341, 1342, 1536]))
    elif which == "decode":
        print(decode_table())
    elif which == "flags":
        print(flags_table())
    elif which == "pin":
        print(pin_table())
    elif which == "jam":
        print(jam_table())
