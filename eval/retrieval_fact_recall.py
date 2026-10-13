#!/usr/bin/env python3
"""Retrieval-layer benchmark: do the i.MX 95 ground-truth facts actually come BACK from RAG?

Measures retrieval on its own, not through the generator or a judge. For each RAG query the
fabricating tasks use, it asks /generate for the retrieved context only (max_tokens=1) and checks
which ground-truth facts appear in the returned chunks. It also asks ChromaDB directly whether each
fact exists ANYWHERE in the knowledge base, so a miss reads as either:
  - not in the KB at all        -> an ingestion/coverage problem, no retriever can fix it
  - in the KB but not retrieved -> a retrieval problem
Facts mirror GROUND_TRUTH in judge_agentic_tasks.py.
"""
import json, os, re, sys, time, requests

BASE = "http://localhost:8080"
CHROMA = "http://localhost:8000/api/v1"
USER, PASSWORD = os.environ.get("SKIPPY_USER", "kyle"), os.environ.get("SKIPPY_PASSWORD", "REDACTED")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/ladder/agentic/retrieval_fact_recall.json")

# fact -> (regex over retrieved text, literal for the KB-wide $contains probe)
FACTS = {
    "Neutron NPU":      (r"Neutron", "Neutron"),
    "2.0 TOPS":         (r"\b2(\.0)?\s*TOPS", "TOPS"),
    "1024 MACs":        (r"\b1024\b", "1024"),
    "Cortex-A55":       (r"Cortex[- ]?A55", "Cortex-A55"),
    "Cortex-M33":       (r"Cortex[- ]?M33", "Cortex-M33"),
    "Cortex-M7":        (r"Cortex[- ]?M7\b", "Cortex-M7"),
    "Mali GPU":         (r"\bMali\b", "Mali"),
    "LPDDR5":           (r"LPDDR5", "LPDDR5"),
    "PCIe Gen3":        (r"PCIe.{0,20}(Gen\s?3|3\.0)", "PCIe"),
    "Ethernet TSN":     (r"\bTSN\b", "TSN"),
    "Energy Flex":      (r"Energy\s*Flex", "Energy Flex"),
}

QUERIES = {
    "email":     ("Write a clear, friendly email to a colleague introducing the NXP i.MX 95 for a new edge-AI "
                  "product. Cover what it is, its Neutron NPU, and why it suits edge inference. Ground the "
                  "specifics in the datasheets.", 4),
    "spec_rag":  ("Using the datasheets, what does the i.MX 95 Neutron NPU do, and what are its key features?", 4),
    "doc_brief": ("Using the datasheets, generate a concise one-page product brief for the NXP i.MX 95: overview, "
                  "NPU, memory/connectivity, and target markets.", 8),
}

def token():
    return requests.post(f"{BASE}/auth/login", json={"username": USER, "password": PASSWORD},
                         timeout=15).json()["token"]

def retrieve(h, prompt, k):
    t0 = time.time()
    d = requests.post(f"{BASE}/generate", headers=h, timeout=1800, json={
        "prompt": prompt, "use_rag": True, "rag_k": k, "max_tokens": 1, "temperature": 0,
        "skip_agent_loop": True}).json()
    ctx = d.get("context_used") or []
    srcs = [c.get("source_file", "?") for c in (d.get("citations") or [])]
    return ctx, srcs, round(time.time() - t0, 2)

def kb_has(literal):
    """Count KB chunks containing `literal` that also mention the i.MX 95 (capped at 200)."""
    cols = requests.get(f"{CHROMA}/collections", timeout=30).json()
    n = 0
    for c in cols:
        r = requests.post(f"{CHROMA}/collections/{c['id']}/get", timeout=120, json={
            "where_document": {"$and": [{"$contains": literal}, {"$contains": "95"}]},
            "include": ["documents"], "limit": 200}).json()
        n += sum(1 for doc in (r.get("documents") or []) if re.search(r"i\.?MX\s?95", doc or ""))
    return n

def main():
    h = {"Authorization": f"Bearer {token()}"}
    kb = {f: kb_has(lit) for f, (_, lit) in FACTS.items()}
    rows = {}
    for task, (prompt, k) in QUERIES.items():
        ctx, srcs, secs = retrieve(h, prompt, k)
        text = "\n".join(ctx)
        hits = {f: bool(re.search(rx, text, re.I)) for f, (rx, _) in FACTS.items()}
        imx95 = sum(1 for c in ctx if re.search(r"i\.?MX\s?95", c))
        rows[task] = {"k": k, "seconds": secs, "n_chunks": len(ctx), "chunks_mentioning_imx95": imx95,
                      "sources": srcs, "facts_retrieved": hits,
                      "recall": round(sum(hits.values()) / len(hits), 3)}
    json.dump({"facts_in_kb_imx95_chunks": kb, "per_query": rows}, open(OUT, "w"), indent=2)

    print(f"\n{'fact':<14} {'in KB':>6}  " + "  ".join(f"{t:>9}" for t in rows))
    for f in FACTS:
        print(f"{f:<14} {kb[f]:>6}  " + "  ".join(f"{('yes' if rows[t]['facts_retrieved'][f] else '--'):>9}" for t in rows))
    print(f"{'RECALL':<14} {'':>6}  " + "  ".join(f"{rows[t]['recall']:>9.0%}" for t in rows))
    for t, r in rows.items():
        print(f"\n[{t}] k={r['k']} {r['n_chunks']} chunks ({r['chunks_mentioning_imx95']} mention i.MX 95), "
              f"{r['seconds']}s\n   sources: {r['sources']}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    sys.exit(main())
