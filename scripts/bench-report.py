#!/usr/bin/env python3
# bench-report.py — aggregate results.csv into median markdown tables (data for RESULTS.md).
# Reps are summarized by MEDIAN (robust to a single slow outlier). Usage: bench-report.py <results.csv>
import csv, sys, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))


def med(xs):
    xs = [float(x) for x in xs if x not in ("", "None", None)]
    return round(st.median(xs), 2) if xs else ""


g = defaultdict(list)
for r in rows:
    g[(r["engine"], r["cell"], r["task"])].append(r)

models = sorted(set(r.get("model", "") for r in rows if r.get("model")))
if models:
    print(f"\n**Model(s):** {', '.join(models)}")
print(f"\n### Per-cell medians ({len(rows)} runs, {len(g)} cells)\n")
hdr = ["cell", "engine", "task", "n", "prompt_tok", "compl_tok", "prefill t/s", "decode t/s",
       "TTFT cold s", "TTFT warm s", "wall s", "correct"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for (eng, cell, task), rs in sorted(g.items()):
    ok = sum(1 for r in rs if r["correct"] == "PASS")
    print("| " + " | ".join(map(str, [
        cell, eng, task, len(rs),
        med([r["prompt_tokens"] for r in rs]), med([r["completion_tokens"] for r in rs]),
        med([r["prefill_tps"] for r in rs]), med([r["decode_tps"] for r in rs]),
        med([r["ttft_s"] for r in rs]), med([r.get("warm_ttft_s", "") for r in rs]),
        med([r["wall_s"] for r in rs]), f"{ok}/{len(rs)}"])) + " |")

# ---- DATA HEALTH: the morning-after trust check ----
npass = sum(1 for r in rows if r["correct"] == "PASS")
nbad = [r for r in rows if r["correct"] not in ("PASS",)]
flagged = [r for r in rows if (r.get("flags") or "").strip()]
print(f"\n### Data health\n")
print(f"- rows captured: **{len(rows)}**  ·  cells: **{len(g)}**  ·  reps/cell: "
      f"{sorted(set(len(v) for v in g.values()))}")
print(f"- correctness PASS: **{npass}/{len(rows)}**")
if nbad:
    print(f"- NON-PASS rows ({len(nbad)}): " + ", ".join(f"{r['cell']}#{r['rep']}={r['correct']}" for r in nbad))
if flagged:
    print(f"- FLAGGED rows ({len(flagged)}) — inspect before trusting:")
    for r in flagged:
        print(f"  - {r['cell']} rep{r['rep']}: {r['flags']}")
else:
    print("- flagged rows: **none** — no multi-instance / low-mem / cache / parse anomalies detected")
