#!/usr/bin/env python3
"""Retrieval benchmark: measure recall@k directly, never through the generator.

WHY THIS EXISTS
---------------
On 2026-07-09 a "−3.97pp INT8 capability loss" turned out to be an artifact. The
causal chain: the retriever missed the evidence → the model correctly said "not in
the excerpts" → a substring grader scored that refusal 0 → we attributed a RETRIEVER
failure to the MODEL. Nobody could see it because the only number anyone measured was
end-to-end answer accuracy.

**Never measure a layer through the layer above it.** This script measures the
retriever alone: given a question whose answer provably exists in the corpus, do the
top-k chunks contain it?

OFFLINE AND READ-ONLY
---------------------
Reads vectors and documents straight out of `vectordb/chroma.sqlite3`. No Docker, no
ChromaDB client, no GPU. Deliberate: the on-disk store was written by Chroma 0.5.0 and
the locally-installed client is 1.5.5, which may attempt a destructive schema
migration. We never let a 1.x client touch that directory.

WHAT IT COMPARES
----------------
Three fusion strategies over identical candidate lists, so any difference is the
fusion and nothing else:

  current   final = 0.7*semantic + 0.3*keyword,  missing scores ZERO-FILLED
            (this is `pipeline/advanced_rag.py::hybrid_search` as shipped)
  balanced  final = 0.5*semantic + 0.5*keyword,  each list min-max normalized,
            missing scores imputed at that list's MINIMUM, not zero
  rrf       Reciprocal Rank Fusion, k=60 (rank-based, immune to missing scores)

The `current` arm is arithmetically incapable of letting a lexical-only chunk win:
its best possible score is 0.3*1.0 = 0.300, while a semantic-only chunk at cosine
0.45 scores 0.7*0.45 = 0.315. See docs/rag-retrieval-diagnosis.md.

    python3 eval/bench_retrieval.py
"""
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from advanced_rag import BM25  # noqa: E402  (exact same class the server uses)

CHROMA = ROOT / "vectordb" / "chroma.sqlite3"
COLLECTION = "personal_knowledge"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # what Chroma 0.5 defaults to

# Ground truth. Each prompt's gold fact provably EXISTS in the corpus (verified by a
# proximity scan, not a substring probe — '3733' alone matches email folder paths,
# '149' matches the app-note number AN14149). See eval/repair_eval_set.py.
GOLD = {
    # --- the 7 the shipped retriever MISSES ---
    "rag_ds_mac_addr34_high_offset": ["410"],
    "rag_ds_rsa_key_size":           ["4096", "RSA"],
    "rag_ds_ecc_curve":              ["P-521"],
    "rag_ds_thermal_rja_11x11":      ["22.5"],
    "rag_imx93_uart":                ["LPUART", "IOMUX"],
    "numprec_lpddr_rate":            ["3733", "MT/s"],
    "multihop_peripheral_count":     ["LPUART", "I2C"],
}


def load_corpus():
    """Documents + vectors for the collection, straight from sqlite. Read-only."""
    con = sqlite3.connect(f"file:{CHROMA}?mode=ro", uri=True)
    # embeddings_queue is a write-ahead log: 7,107 rows for 6,731 distinct ids
    # (376 chunks were re-added). Joining it naively duplicates those chunks and
    # perturbs every ranking. Take the latest row per id.
    rows = con.execute(
        """
        SELECT em.string_value, eq.vector
        FROM embeddings e
        JOIN segments s   ON e.segment_id = s.id
        JOIN collections c ON s.collection = c.id
        JOIN embedding_metadata em ON em.id = e.id AND em.key = 'chroma:document'
        JOIN (SELECT id, vector, MAX(seq_id) FROM embeddings_queue GROUP BY id) eq
             ON eq.id = e.embedding_id
        WHERE c.name = ?
        """,
        (COLLECTION,),
    ).fetchall()
    docs, vecs = [], []
    for text, blob in rows:
        if not text or not blob:
            continue
        n = len(blob) // 4
        v = np.array(struct.unpack(f"<{n}f", blob), dtype=np.float32)
        docs.append(text)
        vecs.append(v)
    V = np.vstack(vecs)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return docs, V


# ---------------- fusion strategies ----------------

def _minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def fuse_current(sem, kw, k):
    """As shipped: 0.7/0.3, zero-fill. Reproduces advanced_rag.hybrid_search's core."""
    kw_n = {i: s / max(kw.values()) for i, s in kw.items()} if kw else {}
    out = {}
    for i in set(sem) | set(kw_n):
        out[i] = 0.7 * sem.get(i, 0.0) + 0.3 * kw_n.get(i, 0.0)
    return sorted(out, key=out.get, reverse=True)[:k]


def fuse_balanced(sem, kw, k):
    """alpha=0.5 convex combination; each list min-max normalized; missing scores
    imputed at that list's MINIMUM (absence of evidence != evidence of absence)."""
    sn, kn = _minmax(sem), _minmax(kw)
    smin = min(sn.values()) if sn else 0.0
    kmin = min(kn.values()) if kn else 0.0
    out = {}
    for i in set(sn) | set(kn):
        out[i] = 0.5 * sn.get(i, smin) + 0.5 * kn.get(i, kmin)
    return sorted(out, key=out.get, reverse=True)[:k]


def fuse_semantic_only(sem, kw, k):
    """What `RAGService.search()` actually does — plain ChromaDB vector query, no
    BM25, no rerank. This is the endpoint `/search` calls, and therefore the
    retriever that produced every frozen RAG context in eval/results/. If this
    arm reproduces the frozen chunks' behaviour, the offline replica is faithful."""
    return sorted(sem, key=sem.get, reverse=True)[:k]


def fuse_rrf(sem, kw, k, rrf_k=60):
    """Rank-based; structurally immune to a missing score."""
    out = {}
    for d in (sem, kw):
        for rank, i in enumerate(sorted(d, key=d.get, reverse=True)):
            out[i] = out.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(out, key=out.get, reverse=True)[:k]


FUSIONS = {"semantic-only (what /search does)": fuse_semantic_only,
           "hybrid 0.7/0.3 zero-fill": fuse_current,
           "hybrid 0.5/0.5 min-impute": fuse_balanced,
           "hybrid RRF (k=60)": fuse_rrf}


def main():
    prompts = {p["id"]: p for p in json.loads((ROOT / "eval" / "prompts_v2.json").read_text())["prompts"]}
    print("loading corpus from sqlite (read-only)…", file=sys.stderr)
    docs, V = load_corpus()
    print(f"  {len(docs)} chunks, dim {V.shape[1]}", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMBED_MODEL)

    # Sanity: does our query encoder match the one that wrote these vectors?
    probe = st.encode([docs[0][:512]], normalize_embeddings=True)[0]
    sim = float(probe @ V[0])
    print(f"  encoder check: re-encoded doc[0] vs stored vector, cosine = {sim:.4f}",
          file=sys.stderr)
    if sim < 0.90:
        print("  ⚠️  encoder mismatch — absolute recall numbers are unreliable, but the",
              file=sys.stderr)
        print("      A/B between fusions still holds (same query vector for all arms).",
              file=sys.stderr)

    print("  fitting BM25 (same class the server uses)…", file=sys.stderr)
    bm = BM25()
    bm.fit(docs)

    KS = [1, 3, 5, 8, 20]
    POOL = 30  # candidate_pool = max(k*2, 30), as in hybrid_search

    results = {name: {k: 0 for k in KS} for name in FUSIONS}
    per_prompt = {}

    for pid, gold in GOLD.items():
        q = prompts[pid]["prompt"]
        qv = st.encode([q], normalize_embeddings=True)[0]
        sims = V @ qv
        top_sem = np.argsort(-sims)[:POOL]
        sem = {int(i): float(sims[i]) for i in top_sem}
        kw = {int(i): float(s) for i, s in bm.search(q, top_k=POOL)}

        per_prompt[pid] = {}
        for name, fn in FUSIONS.items():
            hits = {}
            for k in KS:
                idxs = fn(sem, kw, k)
                # A hit means SOME SINGLE CHUNK carries the fact. Testing the
                # concatenation of top-k lets '4096' come from a byte-size in one
                # chunk and 'RSA' from another — the union accumulates lexical
                # accidents and reports a hit where no chunk contains the answer.
                # That is exactly the failure mode this whole benchmark exists to
                # expose; it would be embarrassing to reproduce it here.
                hit = any(
                    all(g.lower() in docs[i].lower() for g in gold)
                    for i in idxs
                )
                hits[k] = hit
                results[name][k] += int(hit)
            per_prompt[pid][name] = hits

    n = len(GOLD)
    print(f"\nRETRIEVAL RECALL on {n} prompts whose gold fact provably exists in the corpus")
    print("(these are exactly the 7 the shipped retriever misses end-to-end)\n")
    hdr = "fusion".ljust(32) + "".join(f"  R@{k:<4}" for k in KS)
    print(hdr); print("-" * len(hdr))
    for name in FUSIONS:
        row = name.ljust(32)
        for k in KS:
            row += f"  {results[name][k]}/{n}  "
        print(row)

    print("\nper-prompt @k=8:")
    print("  prompt".ljust(34) + "".join(f"{nm[:14]:>16}" for nm in FUSIONS))
    for pid in GOLD:
        row = f"  {pid[:32]:<32}"
        for nm in FUSIONS:
            row += f"{'HIT' if per_prompt[pid][nm][8] else 'miss':>16}"
        print(row)

    out = ROOT / "eval" / "results" / "retrieval_bench.json"
    out.write_text(json.dumps({"n": n, "ks": KS, "totals": results,
                               "per_prompt": per_prompt, "pool": POOL}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
