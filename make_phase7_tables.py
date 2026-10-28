"""Markdown tables for REPORT.md Section 9 from results_watching_crowd/phase7_adaptive.json (stdlib only).

    python3 make_phase7_tables.py {main | zero | jitter | patch | partial}
"""

import json
import os
import sys

S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results_watching_crowd", "phase7_adaptive.json")))
TASKS = (("detection", "Yes", "No"), ("animal", "owl", "cat"))
NAMES = {"body_only": "A: body only", "tail_only": "B: tail only", "kv_pinned": "C: K/V-pinned, end to end",
         "canary_only": "Control: canary only"}
ORDER = ["canary_only_spec", "body_only_spec", "body_only_scaled", "body_only_scaled_lr3e-5", "tail_only_spec_lr3e-5",
         "kv_pinned_spec_lr3e-5", "kv_pinned_scaled", "kv_pinned_scaled_lr3e-5"]


def ci(d):
    return f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"


def label(tag):
    regime = next(r for r in NAMES if tag.startswith(r))
    rest = tag[len(regime) + 1:]
    scale = "512/task" if rest.startswith("spec") else "3,584/task"
    lr = "3e-5" if "lr3e-5" in rest else "5e-6"
    return NAMES[regime], scale, lr


def main_table():
    out = ["| Regime | Training prompts | LR | Steps (best) | Detection: correct / Δ [95% CI] / right way | Animal: correct / Δ [95% CI] / right way | Ordinary KL per position vs original tail, B=1 / B=32 |",
           "|---|---|---|---|---|---|---|"]
    for tag in ORDER:
        r = S["runs"].get(tag)
        if not r:
            continue
        name, scale, lr = label(tag)
        t = r["test"]["adaptive"]
        u = r["utility"]
        tr = r.get("train") or {}
        steps = f"{tr.get('steps', '?')} ({tr.get('best_step', '?')})"
        cells = [f"{t[k]['correct']}/128 / {ci(t[k]['delta'])} / {100 * t[k]['frac_delta_positive']:.0f} %" for k, _, _ in TASKS]
        out.append(f"| {name} | {scale} | {lr} | {steps} | {cells[0]} | {cells[1]} | "
                   f"{u['B1']['kl_per_position']:.4f} / {u['B32']['kl_per_position']:.4f} |")
    return "\n".join(out)


def zero_shot_table():
    out = ["| Condition (held-out, same inputs as the regime) | Original scaled detector: det Δ / animal Δ | Base Qwen3-8B: det Δ / animal Δ |",
           "|---|---|---|"]
    seen = set()
    for tag in ORDER:
        r = S["runs"].get(tag)
        if not r:
            continue
        regime = next(x for x in NAMES if tag.startswith(x))
        if regime in seen:
            continue
        seen.add(regime)
        o, b = r["test"]["original_scaled"], r["test"]["base"]
        out.append(f"| {NAMES[regime]} | {o['detection']['delta']['mean']:+.2f} / {o['animal']['delta']['mean']:+.2f} | "
                   f"{b['detection']['delta']['mean']:+.2f} / {b['animal']['delta']['mean']:+.2f} |")
    return "\n".join(out)


def jitter_table():
    out = ["| Adaptive detector | k = 0 (no jitter): det / animal Δ | k = 1 | k = 2 | k = 4 |", "|---|---|---|---|---|"]
    for tag in ORDER:
        r = S["runs"].get(tag)
        if not r or tag.startswith("canary"):
            continue
        name, scale, lr = label(tag)
        t = r["test"]["adaptive"]
        cells = [f"{t['detection']['delta']['mean']:+.2f} / {t['animal']['delta']['mean']:+.2f} "
                 f"({t['detection']['correct'] + t['animal']['correct']}/256)"]
        for k in ("1", "2", "4"):
            j = r["jitter"][k]
            cells.append(f"{j['detection']['delta']['mean']:+.2f} / {j['animal']['delta']['mean']:+.2f} "
                         f"({j['detection']['correct'] + j['animal']['correct']}/256)")
        out.append(f"| {name}, {scale}, lr {lr} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def patch_table():
    out = []
    for tag in ORDER:
        r = S["runs"].get(tag)
        if not r or "patching" not in r:
            continue
        name, scale, lr = label(tag)
        p = r["patching"]
        if p["detection"].get("signal_too_small"):
            continue
        out.append(f"\n**{name}, {scale}, lr {lr}** (replay with the tail at [1, L]; mean shift to restore: "
                   f"det {p['detection']['mean_total_shift_solo_geometry']:+.2f}, animal {p['animal']['mean_total_shift_solo_geometry']:+.2f} nats; "
                   "– = shift under 0.5 nats, nothing to attribute)\n")

        def cell(q, f):
            if q.get("signal_too_small"):
                return "–"
            v = f(q)
            return "–" if v is None or v != v else f"{100 * v:.0f} %"
        out.append("| Patched from the B=32 input into the B=1 run | Detection | Animal |")
        out.append("|---|---|---|")
        rows = [("whole prompt body", lambda q: q["groups"]["body_all"])]
        rows += [(f"body quintile {i}", lambda q, i=i: q["groups"][f"body_q{i}"]) for i in range(1, 6)]
        rows += [("last body token (L−10)", lambda q: q["groups"]["body_last_token"])]
        if tag.startswith("body_only"):
            out.append("| 12 template positions (0–2, L−9…L−1) | identical by construction (overwritten with solo values) | same |")
        else:
            rows = [("positions 0–2 (prefix)", lambda q: q["groups"]["prefix_0_2"]),
                    ("all 9 suffix tokens", lambda q: q["groups"]["suffix_all"])] + rows
            rows += [(f"suffix `{k}`", lambda q, k=k: q["suffix"][k]) for k in p["detection"]["suffix"]]
        rows += [("best single body token (mean over prompts)", lambda q: q["body_single_max_mean"])]
        for lab, f in rows:
            out.append(f"| {lab} | {cell(p['detection'], f)} | {cell(p['animal'], f)} |")
        for comp in ("attn34", "mlp34", "attn35", "mlp35"):
            out.append(f"| component `{comp}` | {cell(p['detection'], lambda q: q['components'][comp])} | "
                       f"{cell(p['animal'], lambda q: q['components'][comp])} |")
        for li in ("34", "35"):
            cells = []
            for t, _, _ in TASKS:
                if p[t].get("signal_too_small"):
                    cells.append("–")
                    continue
                v = p[t]["heads"][li]
                top = sorted(range(32), key=lambda h: -v[h])[:3]
                cells.append(", ".join(f"H{h} {100 * v[h]:.0f} %" for h in top))
            out.append(f"| top heads, layer {li} | {cells[0]} | {cells[1]} |")
        f = p["detection"]["factorial"]
        out.append(f"\nFactorial (mean det log-odds): input B=1 / tail [1, L] {f['solo_in_solo']:+.2f}; input B=32 / [1, L] "
                   f"{f['crowd_in_solo']:+.2f}; input B=1 / [32, L] {f['solo_in_crowd']:+.2f}; input B=32 / [32, L] {f['crowd_in_crowd']:+.2f}.")
    return "\n".join(out)


def partial_table():
    out = ["| Run stopped by the spend limit (not rerun) | Last step | Validation: det both / Δ | Validation: animal both / Δ |",
           "|---|---|---|---|"]
    for tag, r in S.get("partial_runs", {}).items():
        name, scale, lr = label(tag)
        v = r["val"]
        out.append(f"| {name}, {scale}, lr {lr} | {r['last_step']} (eval at {r['last_eval_step']}) | "
                   f"{v['detection']['both']}/64 / {v['detection']['mean_delta']:+.2f} | {v['animal']['both']}/64 / {v['animal']['mean_delta']:+.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print({"main": main_table, "zero": zero_shot_table, "jitter": jitter_table, "patch": patch_table,
           "partial": partial_table}[sys.argv[1]]())
