#!/usr/bin/env python3
"""Eval-set integrity checker. Stdlib only. Run this BEFORE trusting any pass rate.

Found 2026-07-09 after a −3.97pp "INT8 capability loss" turned out to be an
artifact of the eval set rather than of quantization. Three defect classes, all
of which make a *substring* pass rate (and any LLM judge anchored to the same
gold strings) measure lexical accidents instead of capability:

  1. EMPTY gold_substrings
     The grader returns "manual" — the prompt can never pass, but it still sits
     in the denominator. A constant negative offset that differentiates nothing.

  2. GOLD FACT ABSENT FROM THE FROZEN RAG CONTEXT
     The retriever never surfaced the chunk containing the answer, so the model
     *cannot* answer from context. Correct behaviour is to refuse — and a refusal
     scores 0 unless it happens to contain a gold token. Worked example:
       rag_ds_imx93_149_gpio, gold=["i.MX 93"], context contains no "149" at all.
       fp16 : "...no explicit mention of ... exactly 149 GPIO pins. The information
               provided focuses on ... the i.MX 93 processors ..."   -> PASS
       int8 : "...no specific mention of ... exactly 149 GPIO pins. The information
               provided focuses on the pin counts for different package types..."  -> FAIL
     Two identical refusals. The delta is whether a chip name appeared in a
     subordinate clause. This single prompt supplied 2 of the 7 "regressions"
     that were briefly published as an INT8 capability loss.

  3. WEAK GOLD
     The gold token occurs so often in its own retrieved context (e.g. "MHz" 82×)
     that almost any fluent sentence contains it. Near-zero discriminative power.

Exit code 1 if any defect is found, so this can gate a CI run.

    python3 check_eval_integrity.py --prompts prompts_v2.json --rag-chunks rag_chunks_for_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

WEAK_GOLD_THRESHOLD = 20   # occurrences of the gold token in its own context


def audit(prompts_path: Path, chunks_path: Path):
    prompts = json.loads(prompts_path.read_text())
    prompts = prompts["prompts"] if isinstance(prompts, dict) else prompts
    chunks = json.loads(chunks_path.read_text()).get("chunks_by_id", {})

    empty, unanswerable, weak = [], [], []

    declared = []
    for p in prompts:
        pid = p["id"]
        # A prompt explicitly quarantined from the headline is not a defect —
        # it's a documented exclusion, same as `persona`. Report it, don't fail.
        if p.get("category_status") == "BROKEN_SUBSTRING_INCOMPATIBLE":
            declared.append(pid)
            continue
        gold = p.get("gold_substrings") or []
        if not gold:
            empty.append(pid)
            continue
        if p.get("match_mode") == "any":
            continue  # refusal prompts: gold is a list of refusal phrasings, not facts
        ctx = " ".join(
            (c.get("text", "") if isinstance(c, dict) else str(c))
            for c in chunks.get(pid, [])
        ).lower()
        if not ctx:
            continue  # non-retrieval prompt; nothing to check against
        missing = [g for g in gold if g.lower() not in ctx]
        if missing:
            unanswerable.append((pid, missing))
            continue
        # With match_mode="all" EVERY gold token must appear, so the RAREST one
        # carries the discrimination. Flagging a prompt because *some* token is
        # common over-reports: rag_ds_package_pitch has gold ['0.5','mm'] and
        # '0.5' occurs exactly once — that prompt is fine. Only flag when even
        # the rarest gold token is common enough to be hit by any fluent answer.
        counts = {g: ctx.count(g.lower()) for g in gold}
        if min(counts.values()) >= WEAK_GOLD_THRESHOLD:
            weak.append((pid, counts))

    return prompts, empty, unanswerable, weak, declared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts_v2.json")
    ap.add_argument("--rag-chunks", default="rag_chunks_for_v2.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    prompts, empty, unans, weak, declared = audit(here / a.prompts, here / a.rag_chunks)

    print(f"Eval-set integrity audit — {len(prompts)} prompts\n")

    print(f"0. DECLARED EXCLUSIONS (quarantined, not scored — not a defect): {len(declared)}")
    for pid in declared:
        print(f"     {pid}")
    print()

    print(f"1. EMPTY gold_substrings (can never pass; still in denominator): {len(empty)}")
    for pid in empty:
        print(f"     {pid}")

    print(f"\n2. GOLD FACT ABSENT from frozen RAG context (unanswerable): {len(unans)}")
    for pid, miss in unans:
        print(f"     {pid:<34} missing {miss}")

    print(f"\n3. WEAK GOLD (EVERY gold token appears >={WEAK_GOLD_THRESHOLD}x in its own context): {len(weak)}")
    for pid, counts in weak:
        print(f"     {pid:<34} {counts}")

    affected = set(empty) | {u[0] for u in unans} | {w[0] for w in weak}
    pct = len(affected) / len(prompts) if prompts else 0
    print(f"\nTOTAL prompts with >=1 defect: {len(affected)} / {len(prompts)} = {pct:.0%}")

    if affected:
        print("\n⚠️  DO NOT quote a pass rate over the full set without excluding these.")
        print("    A substring grader — and any LLM judge anchored to the same gold")
        print("    strings — will score lexical accidents, not capability. A model that")
        print("    correctly refuses an unanswerable question scores 0.")
        print("\n    Deltas BETWEEN models on defective prompts are especially treacherous:")
        print("    two near-identical refusals can differ by whether one happened to name")
        print("    the chip. That is how a 0.0pp INT8 result was published as -3.97pp.")
        return 1
    print("\n✓ no defects found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
