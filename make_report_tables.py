"""Render the markdown tables used in REPORT.md Sections 5-6 from results_watching_crowd/*.json (stdlib only).

    python3 make_report_tables.py [all | main | sweep | vllm | mech]
"""

import json
import os
import random
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_watching_crowd")
TASKS = (("detection", "Yes", "No", "Detection (Yes/No)"), ("animal", "owl", "cat", "Animal (owl/cat)"))
SWEEP = ["1", "2", "4", "8", "16", "24", "32", "48", "64"]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x):
    return f"{x['mean']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)"


def boot(vals, n=10000, seed=42):
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    ms = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return {"mean": m, "ci95": [ms[int(0.025 * n)], ms[int(0.975 * n)]]}


def main_table(tags):
    out = ["| Model | Task | Subset | Correct answers (B=1 & B=32) | Both right (pairs) | "
           "B=1 → target | B=32 → target | Δ log-odds, nats [95% CI] | Δ > 0 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for tag, label in tags:
        s = load(f"summary_{tag}.json")
        if not s:
            continue
        for model in ("trained", "base"):
            if model not in s or (model == "base" and tag != tags[0][0]):
                continue
            name = label if model == "trained" else "Base Qwen3-8B"
            for task, pos, neg, tname in TASKS:
                for sub in ("all", "seen_families", "new_families"):
                    x = s[model][task][sub]
                    if x is None:
                        continue
                    out.append(
                        f"| {name} | {tname} | {sub.replace('_', ' ')} ({x['n_prompts']}) | "
                        f"{pct(x['correct_answers'], x['answers'])} | {x['both_correct_pairs']}/{x['n_prompts']} | "
                        f"{x['solo_correct']}/{x['n_prompts']} {neg} | {x['crowd_correct']}/{x['n_prompts']} {pos} | "
                        f"{ci(x['delta_log_odds'])} | {100 * x['frac_delta_positive']:.0f}% |")
    return "\n".join(out)


def sweep_table(tag):
    s = load(f"summary_{tag}.json")
    if not s:
        return ""
    head = "| B | " + " | ".join(SWEEP) + " |"
    rows = [head, "|---" * (len(SWEEP) + 1) + "|"]
    for model in ("trained", "base"):
        if model not in s:
            continue
        for task, pos, neg, tname in TASKS:
            sw = s[model]["sweep"][task]
            rows.append(f"| {model}: says {pos} (of 64) | " + " | ".join(str(sw[b]["says_crowded"]) for b in SWEEP) + " |")
            rows.append(f"| {model}: mean {pos}−{neg} log-odds | " + " | ".join(f"{sw[b]['log_odds']['mean']:.2f}" for b in SWEEP) + " |")
    if "perturbation" in s:
        p = s["perturbation"]
        rows.append("| layer-34 input: rel. L2 vs B=1 | " + " | ".join(f"{p[b]['rel_l2_mean']:.4f}" for b in SWEEP) + " |")
        rows.append("| layer-34 input bit-identical to B=32 | " + " | ".join(f"{100 * p[b]['frac_bit_identical_to_B32']:.0f}%" for b in SWEEP) + " |")
    return "\n".join(rows)


def vllm_table():
    out = ["| Model served by vLLM 0.19.1 | Task | B=1 → target | B=32 → target | Δ log-odds [95% CI] | Δ > 0 |",
           "|---|---|---|---|---|---|"]
    for tag in ("base", "main", "scaled"):
        v = load(f"vllm_transfer_{tag}_test.json")
        if not v:
            continue
        for task, pos, neg, tname in TASKS:
            rs = [r for r in v["rows"] if r["task"] == task]
            d = [r["B"]["32"]["log_odds"] - r["B"]["1"]["log_odds"] for r in rs]
            out.append(f"| {tag} | {tname} | {sum(r['B']['1']['lenient'] == 'neg' for r in rs)}/64 {neg} | "
                       f"{sum(r['B']['32']['lenient'] == 'pos' for r in rs)}/64 {pos} | {ci(boot(d))} | "
                       f"{100 * sum(x > 0 for x in d) / len(d):.0f}% |")
    return "\n".join(out)


def mech_table(tag):
    s = load(f"summary_{tag}.json")
    if not s or "mechanism" not in s:
        return ""
    m = s["mechanism"]
    out = ["| Task | input B=1, tail B=1 | input B=32, tail B=1 | input B=1, tail B=32 | input B=32, tail B=32 |",
           "|---|---|---|---|---|"]
    for task, pos, neg, tname in TASKS:
        f = m["factorial"][task]
        out.append(f"| {tname} | " + " | ".join(ci(f[k]) for k in (
            "solo_input_solo_geom", "crowd_input_solo_geom", "solo_input_crowd_geom", "crowd_input_crowd_geom")) + " |")
    out.append("")
    out.append("| Patched position (tail at B=1) | " + " | ".join(t[3] for t in TASKS) + " |")
    out.append("|---|---|---|")
    keys = list(m["positions"][f"{TASKS[0][0]}_solo_geom"].keys())
    for k in keys:
        out.append(f"| `{k}` | " + " | ".join(f"{100 * m['positions'][f'{t[0]}_solo_geom'][k]:.0f}%" for t in TASKS) + " |")
    out.append("")
    out.append("| Patched component (tail at B=1) | " + " | ".join(t[3] for t in TASKS) + " |")
    out.append("|---|---|---|")
    for k in m["components"][f"{TASKS[0][0]}_solo_geom"]:
        out.append(f"| {k} | " + " | ".join(f"{100 * m['components'][f'{t[0]}_solo_geom'][k]:.0f}%" for t in TASKS) + " |")
    out.append("")
    out.append("| Top heads (all positions, tail at B=1) | " + " | ".join(t[3] for t in TASKS) + " |")
    out.append("|---|---|---|")
    for layer in ("34", "35"):
        cells = []
        for t in TASKS:
            v = m["heads"][f"{t[0]}_L{layer}_all_solo_geom"]
            top = sorted(range(32), key=lambda h: -v[h])[:3]
            cells.append(", ".join(f"H{h} {100 * v[h]:.0f}%" for h in top))
        out.append(f"| layer {layer} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tags = [("scaled", "Trained (scaled, 3,584 prompts/task)"), ("main", "Trained (spec, 512 prompts/task)")]
    if which in ("all", "main"):
        print(main_table(tags))
    if which in ("all", "sweep"):
        for tag, _ in tags:
            print(f"\n**Sweep — {tag}**\n")
            print(sweep_table(tag))
    if which in ("all", "vllm"):
        print()
        print(vllm_table())
    if which in ("all", "mech"):
        for tag, _ in tags:
            print(f"\n**Mechanism — {tag}**\n")
            print(mech_table(tag))
