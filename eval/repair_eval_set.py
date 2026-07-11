#!/usr/bin/env python3
"""Repair prompts_v2 → prompts_v2_1, with evidence for every change.

Motivation (2026-07-09). A "−3.97pp INT8 capability loss" turned out to be an
artifact of this eval set. Auditing it against the *actual* ChromaDB corpus
(`vectordb/chroma.sqlite3`, 6,731 chunks) found three distinct defects:

  A. 7 prompts whose gold fact IS in the corpus but which the k=8 hybrid
     retriever never surfaced. Example: "RSA up to 4096, ECC curves up to P-521"
     sits in 13 and 10 chunks respectively, and the frozen context for
     rag_ds_rsa_key_size / rag_ds_ecc_curve contains neither. These were scored
     as MODEL failures. They are RETRIEVER failures.

  B. 6 prompts whose gold fact is **not in the knowledge base at all**. The two
     GPIO prompts even assert contradictory golds (148 vs 149); the only corpus
     hits for those digits are the page number "1480 / 5652" and the application
     note "AN14149". A perfect RAG model scores zero on all six.

  C. Weak golds: every gold token so common in its own context that any fluent
     answer contains it ('i.MX 93' × 100, 'LPDDR4' × 49).

Repairs:

  A → **oracle chunks.** Splice the chunk that actually contains the fact in at
      rank 1 and drop the lowest-ranked chunk, holding k=8. Labelled
      `retrieval: "oracle"`. This is what the corpus already *claimed* to do:
      freeze retrieval so any difference is attributable to the model and never
      to the retriever. Oracle retrieval is the cleanest way to isolate the model.
      It measures reading comprehension, NOT end-to-end RAG. Do not conflate.

  B → **hallucination probes.** `match_mode: "any"`, gold = refusal phrasings,
      category `faithfulness`. Refusing is now correct, because the context
      genuinely cannot support an answer. Asserting the fact is parametric
      bleed-through, which for a RAG assistant is the failure mode of interest.
      These are *harder* than the existing `refusal` prompts because the chips
      are real.

  C → tightened golds ('A55' → 'Cortex-A55'; '1.7'+'GHz' → '1.7 GHz';
      'LPDDR4' → '3733'+'MT/s', which is the answer the question actually asks for).

Every edit is emitted with the evidence that justified it. Nothing is silent.

    python3 eval/repair_eval_set.py [--dry-run]
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROMA = ROOT / "vectordb" / "chroma.sqlite3"
PROMPTS_IN = ROOT / "eval" / "prompts_v2.json"
CHUNKS_IN = ROOT / "eval" / "results" / "rag_chunks_for_v2.json"
PROMPTS_OUT = ROOT / "eval" / "prompts_v2_1.json"
CHUNKS_OUT = ROOT / "eval" / "results" / "rag_chunks_for_v2_1.json"

REFUSAL_GOLD = [
    "don't", "doesn't", "does not", "do not", "can't", "cannot", "unable",
    "unknown", "not aware", "no information", "no record", "no mention",
    "haven't", "not familiar", "no data", "not contain", "not specify",
    "not specified", "no specific", "not provided", "not explicitly",
]

# (prompt_id, anchor_token, co_token) — the fact is the two appearing together.
ORACLE = {
    "rag_ds_mac_addr34_high_offset": ("MAC_ADDRESS34_HIGH", "410"),
    "rag_ds_rsa_key_size":           ("4096", "RSA"),
    "rag_ds_ecc_curve":              ("P-521", "ECC"),
    "rag_ds_thermal_rja_11x11":      ("22.5", "thermal"),
    "rag_imx93_uart":                ("LPUART", "IOMUX"),
    "numprec_lpddr_rate":            ("3733", "MT/s"),
    "multihop_peripheral_count":     ("LPUART", "I2C"),
}

# Fact provably absent from the 6,731-chunk corpus → becomes a hallucination probe.
UNANSWERABLE = {
    "rag_ds_imx93_149_gpio":     "no chunk states an i.MX 93 GPIO count; only hit for '149' is the app-note number AN14149",
    "rag_ds_imx93_gpio_count":   "no chunk states an i.MX 93 GPIO count; only hit for '148' is the page number '1480 / 5652'",
    "rag_ds_eiq_neutron_npu":    "the string 'Neutron' appears in 0 of 6731 chunks",
    "multihop_cpu_plus_npu":     "the string 'Neutron' appears in 0 of 6731 chunks",
    "multihop_imx93_vs_95_cores":"6 chunks mention 'i.MX 95', none state its Cortex-A core count",
    "rag_angry_birds_demo":      "'Angry Birds' appears only in a demo video transcript; no chip is named near it",
}

# Weak golds → discriminating replacements.
TIGHTEN = {
    "rag_imx93_cortex":               ["Cortex-A55", "Cortex-M33"],
    "rag_ds_imx93_cortex_a_max_freq": ["1.7 GHz"],
    "numprec_lpddr_rate":             ["3733", "MT/s"],   # question asks for the RATE, not the DRAM type
}

# Empty gold + subjective → cannot be substring-graded, quarantine like persona.
QUARANTINE = {
    "general_embedded_book": "empty gold_substrings; a book recommendation cannot be substring-graded",
}


def load_corpus():
    con = sqlite3.connect(CHROMA)
    rows = [r[0] for r in con.execute(
        "select string_value from embedding_metadata "
        "where key='chroma:document' and string_value is not null")]
    return rows


def find_oracle_chunk(corpus, anchor, co, window=140):
    """Best chunk containing `anchor` within `window` chars of `co`.
    Ranked by how tight the co-occurrence is — tighter means the two tokens are
    part of the same statement rather than coincidentally in the same page."""
    best, best_gap = None, 10**9
    al, cl = anchor.lower(), co.lower()
    for doc in corpus:
        dl = doc.lower()
        for m in re.finditer(re.escape(al), dl):
            lo, hi = max(0, m.start() - window), m.start() + len(al) + window
            seg = dl[lo:hi]
            j = seg.find(cl)
            if j == -1:
                continue
            gap = abs((lo + j) - m.start())
            if gap < best_gap:
                best, best_gap = doc, gap
    return best, best_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    corpus = load_corpus()
    print(f"corpus: {len(corpus)} chunks\n")

    doc = json.loads(PROMPTS_IN.read_text())
    prompts = doc["prompts"]
    chunks = json.loads(CHUNKS_IN.read_text())
    by_id = chunks["chunks_by_id"]

    changes = []

    for p in prompts:
        pid = p["id"]

        # --- C: tighten weak golds (do this before A so numprec gold is right)
        if pid in TIGHTEN:
            old = list(p["gold_substrings"])
            p["gold_substrings"] = TIGHTEN[pid]
            changes.append(("tighten-gold", pid, f"{old} -> {p['gold_substrings']}"))

        # --- A: oracle chunk splice
        if pid in ORACLE:
            anchor, co = ORACLE[pid]
            chunk, gap = find_oracle_chunk(corpus, anchor, co)
            if chunk is None:
                changes.append(("ORACLE-FAILED", pid, f"no chunk with '{anchor}' near '{co}'"))
                continue
            ctx = by_id.get(pid, [])
            # verify every gold token is now present
            merged = [{"text": chunk, "source": "oracle"}] + ctx[:-1]
            joined = " ".join(c["text"] for c in merged).lower()
            missing = [g for g in p["gold_substrings"] if g.lower() not in joined]
            if missing:
                changes.append(("ORACLE-INSUFFICIENT", pid, f"still missing {missing}"))
                continue
            by_id[pid] = merged
            changes.append(("oracle-chunk", pid,
                            f"'{anchor}' within {gap} chars of '{co}'; k={len(merged)}"))

        # --- B: unanswerable -> hallucination probe
        if pid in UNANSWERABLE:
            p["category"] = "faithfulness"
            p["match_mode"] = "any"
            p["gold_substrings"] = list(REFUSAL_GOLD)
            p["unanswerable_from_context"] = True
            p["evidence"] = UNANSWERABLE[pid]
            changes.append(("hallucination-probe", pid, UNANSWERABLE[pid]))

        # --- quarantine
        if pid in QUARANTINE:
            p["category_status"] = "BROKEN_SUBSTRING_INCOMPATIBLE"
            p["quarantine_reason"] = QUARANTINE[pid]
            changes.append(("quarantine", pid, QUARANTINE[pid]))

    print(f"{'action':<22} {'prompt':<32} evidence")
    print("-" * 110)
    for act, pid, ev in changes:
        print(f"{act:<22} {pid:<32} {ev[:52]}")

    failures = [c for c in changes if c[0].startswith("ORACLE-")]
    print(f"\n{len(changes)} changes, {len(failures)} failures")

    doc["version"] = "2.1"
    doc["prompts"] = prompts
    doc["repaired"] = "2026-07-09"
    doc["repair_notes"] = (
        "Oracle chunks spliced for 7 prompts whose gold fact was in the corpus but "
        "not retrieved (retrieval='oracle' — this measures model comprehension, NOT "
        "end-to-end RAG). 6 prompts whose gold fact is absent from the corpus "
        "converted to hallucination probes (category='faithfulness', match_mode='any'). "
        "Weak golds tightened. See eval/repair_eval_set.py for evidence."
    )
    chunks["chunks_by_id"] = by_id
    chunks["retrieval"] = "hybrid_k8 + oracle splice for 7 prompts"

    if a.dry_run:
        print("\n(dry run — nothing written)")
        return 0 if not failures else 1

    PROMPTS_OUT.write_text(json.dumps(doc, indent=2))
    CHUNKS_OUT.write_text(json.dumps(chunks, indent=2))
    print(f"\nwrote {PROMPTS_OUT}")
    print(f"wrote {CHUNKS_OUT}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
