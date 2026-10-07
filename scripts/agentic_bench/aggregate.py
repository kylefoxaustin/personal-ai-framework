#!/usr/bin/env python3
"""Aggregate the latest N trace JSONs per task into stable means for the deck."""
import json, glob, statistics
TASKS = ["decode_heavy", "prefill_heavy", "rag_multistep", "field_service", "inbox_triage"]
N = 3
def mean(x): return round(statistics.mean(x), 1) if x else 0
rows = []
for t in TASKS:
    files = sorted(glob.glob(f"eval/results/ladder/agentic/trace_{t}_*.json"))[-N:]
    wall, pf, dc, dtps, smp, memp, pwr = ([] for _ in range(7))
    for f in files:
        d = json.load(open(f)); te = d.get("telemetry", {}); g = d.get("gpu_signature", {})
        wall.append(d.get("wall_s", 0)); pf.append(te.get("prefill_ms") or 0)
        dc.append(te.get("decode_ms") or 0); dtps.append(te.get("decode_tok_per_s") or 0)
        smp.append(g.get("sm_pct", {}).get("mean", 0)); memp.append(g.get("mem_pct", {}).get("mean", 0))
        pwr.append(g.get("power_w", {}).get("mean", 0))
    llm = mean(pf) + mean(dc)
    share = round(mean(dc) / llm * 100, 1) if llm else 0
    orch = round((mean(wall) * 1000 - llm) / (mean(wall) * 1000) * 100, 1) if mean(wall) else 0
    rows.append((t, len(files), mean(wall), mean(pf), mean(dc), share, orch, mean(dtps), mean(smp), mean(memp), mean(pwr)))

hdr = ["task", "n", "wall_s", "pf_ms", "dc_ms", "dec%LLM", "orch%task", "dec_tps", "sm%", "mem%", "pwr_W"]
w = [15, 3, 7, 8, 8, 8, 10, 8, 6, 6, 7]
print(" ".join(h.ljust(wi) for h, wi in zip(hdr, w)))
print("-" * (sum(w) + len(w)))
for r in rows:
    print(" ".join(str(v).ljust(wi) for v, wi in zip(r, w)))
