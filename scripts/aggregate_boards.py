#!/usr/bin/env python3
"""Aggregate the per-board runs from bench_board.py into one decode ladder.

Reads eval/results/ladder/board_runs/*.json (written by bench_board.py) and prints the
comparable decode/prefill ladder, each rung tagged by backend + provenance. Writes a combined
eval/results/ladder/board_runs/_ladder.json. Run after benching the boards you care about.
"""
import glob, json, os
REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
D = os.path.join(REPO, "eval/results/ladder/board_runs")
ORDER = ["thor", "orin", "iq9", "imx95-cpu", "imx95-ara"]  # 5090 is the full-stack reference, separate

def load():
    out = {}
    for f in glob.glob(os.path.join(D, "*.json")):
        b = os.path.basename(f)[:-5]
        if b.startswith("_"): continue
        out[b] = json.load(open(f))
    return out

def main():
    runs = load()
    rows = []
    for b in ORDER:
        r = runs.get(b)
        if not r: continue
        res = r.get("result", {})
        rows.append((b, res.get("decode_tok_s"), res.get("prefill_tok_s"),
                     res.get("backend", r.get("backend")),
                     res.get("decode_prov", "MEASURED" if r.get("backend","").startswith(("cuda","cpu")) else "")))
    print(f"\n{'board':12s} {'decode t/s':>10s} {'prefill t/s':>11s}  {'backend':14s} provenance")
    print("-"*72)
    print(f"{'RTX 5090':12s} {217.9:>10} {'(full-stack)':>11s}  {'cuda':14s} MEASURED (reference, n=3)")
    for b, dc, pf, be, prov in rows:
        print(f"{b:12s} {str(dc):>10s} {str(pf):>11s}  {be:14s} {prov or 'MEASURED'}")
    comb = {"ladder": [{"board": b, "decode_tok_s": dc, "prefill_tok_s": pf, "backend": be, "prov": prov}
                       for b, dc, pf, be, prov in rows],
            "reference": {"board": "rtx5090", "decode_tok_s": 217.9, "prov": "MEASURED n=3"},
            "note": "7B-class Q4 decode. Cross-instrument (5090 full-stack telemetry; Jetsons+i.MX95 llama-bench; "
                    "iq9 Genie [qualcomm]; imx95-ara Kinara .dvm dated). Within-board size ratios are the clean comparison."}
    json.dump(comb, open(os.path.join(D, "_ladder.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(D,'_ladder.json')}")

if __name__ == "__main__":
    main()
