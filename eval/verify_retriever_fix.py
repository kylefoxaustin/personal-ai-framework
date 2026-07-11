#!/usr/bin/env python3
"""Drive the REAL retriever code against the REAL Chroma index and measure recall@8.

Unlike eval/bench_retrieval.py (which reimplements fusion over raw vectors), this
imports pipeline/rag_service.py and pipeline/advanced_rag.py and calls them exactly
as the server does. It is the ground-truth verifier for the retriever fixes: run it
before and after each change and watch the number move.

Needs the vectordb container up on localhost:8000. No GPU.

    python3 eval/verify_retriever_fix.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from rag_service import RAGService
from advanced_rag import AdvancedRAG

# The 7 prompts whose gold fact provably exists in the corpus (proximity-verified,
# eval/repair_eval_set.py). numprec gold corrected to the actual answer.
GOLD = {
    "rag_ds_mac_addr34_high_offset": ["410"],
    "rag_ds_rsa_key_size":           ["4096", "RSA"],
    "rag_ds_ecc_curve":              ["P-521"],
    "rag_ds_thermal_rja_11x11":      ["22.5"],
    "rag_imx93_uart":                ["LPUART", "IOMUX"],
    "numprec_lpddr_rate":            ["3733", "MT/s"],
    "multihop_peripheral_count":     ["LPUART", "I2C"],
}


def chunk_hit(texts, gold):
    """A hit means SOME SINGLE chunk carries every gold token. Never the union —
    that is the lexical-accident trap that produced the phantom INT8 result."""
    return any(all(g.lower() in t.lower() for g in gold) for t in texts)


def main():
    prompts = {p["id"]: p for p in
               json.loads((ROOT / "eval" / "prompts_v2.json").read_text())["prompts"]}
    rag = RAGService(chroma_host="localhost", chroma_port=8000,
                     collection_name="personal_knowledge")
    adv = AdvancedRAG(rag)

    md = rag.collection.metadata or {}
    print(f"collection metadata: {md}")
    print(f"collection count:    {rag.collection.count()}\n")

    sem_hits = hyb_hits = 0
    print(f"{'prompt':<32}{'semantic /search':>18}{'hybrid (chat)':>16}")
    print("-" * 66)
    for pid, gold in GOLD.items():
        q = prompts[pid]["prompt"]
        sem = [r["content"] for r in rag.search(q, k=8)]
        hyb = [r.content for r in adv.hybrid_search(q, k=8)]
        s, h = chunk_hit(sem, gold), chunk_hit(hyb, gold)
        sem_hits += s; hyb_hits += h
        print(f"{pid[:32]:<32}{('HIT' if s else 'miss'):>18}{('HIT' if h else 'miss'):>16}")

    n = len(GOLD)
    print(f"\n  semantic (/search path) : {sem_hits}/{n} = {sem_hits/n:.0%}")
    print(f"  hybrid   (chat path)    : {hyb_hits}/{n} = {hyb_hits/n:.0%}")
    return sem_hits, hyb_hits


if __name__ == "__main__":
    main()
