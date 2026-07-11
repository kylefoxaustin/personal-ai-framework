#!/usr/bin/env python3
"""Re-gather frozen RAG contexts through the FIXED hybrid retriever → v3.

Runs the actual AdvancedRAG.hybrid_search against the live Chroma index (no GPU,
no llm-server, no auth). Produces rag_chunks_for_v3.json in the same schema as
rag_chunks_for_v2_1.json.

v2.1 vs v3 — two legitimately different eval configs, keep both:
  v2.1  oracle chunks hand-spliced   → retrieval held constant, isolates the MODEL
        (this is what [qualcomm] ran his Q8_0 quantizer comparison on — do not touch)
  v3    real hybrid retrieval         → measures the END-TO-END PRODUCT
        (the retriever fixes of 2026-07-10 make this the product's real behaviour)

Needs the vectordb container up and a chromadb==0.5.0 client (numpy<2).

    python3 eval/regather_hybrid_v3.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from rag_service import RAGService
from advanced_rag import AdvancedRAG

K = 8


def main():
    prompts = json.loads((ROOT / "eval" / "prompts_v2_1.json").read_text())["prompts"]
    rag = RAGService(chroma_host="localhost", chroma_port=8000,
                     collection_name="personal_knowledge")
    adv = AdvancedRAG(rag)

    out = {
        "source_prompt_set": "eval/prompts_v2_1.json",
        "k": K,
        "retrieval": ("hybrid_search: BM25 + semantic (over-fetch pool max(k,32)) + "
                      "min-impute 0.5/0.5 fusion + rerank — product-faithful, "
                      "retriever fixes 2026-07-10"),
        "chunks_by_id": {},
    }

    for p in prompts:
        pid = p["id"]
        # Only RAG-grounded categories get retrieved context, matching the frozen set.
        hits = adv.hybrid_search(p["prompt"], k=K)
        out["chunks_by_id"][pid] = [
            {"text": h.content,
             "source": (h.metadata or {}).get("source_file") or (h.metadata or {}).get("source"),
             "score": h.final_score}
            for h in hits
        ]
        print(f"  {pid:<34} {len(hits)} chunks", file=sys.stderr)

    dest = ROOT / "eval" / "results" / "rag_chunks_for_v3.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
