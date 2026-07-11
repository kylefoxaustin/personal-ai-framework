# Skippy's retriever: diagnosis, literature, and a measured plan

**Date:** 2026-07-09
**Status:** FIXES APPLIED AND MEASURED against the live index (2026-07-10)
**Measured:** recall@8 = **20/27 = 74%** (per-chunk containment) on datasheet questions whose
gold fact provably exists in the corpus. **Three separate defects, decomposed below.**

---

## 0. How we got here

A "−3.97pp INT8 capability loss" was published to three sessions. It was an artifact.
Tracing it produced a chain in which **every layer was a reasonable default and no
layer measured the one beneath it**:

```
  weak embedder + zero-filled fusion
        ↓  retriever never surfaces the evidence
  model correctly answers "not in the excerpts"
        ↓  substring grader scores the refusal 0
           (unless the refusal happens to name the chip)
  "INT8 costs -3.97pp of task accuracy"
        ↓  three sessions nearly ship it
```

A green number at the top looked like evidence for everything below it.

---

## 1. THREE defects, measured and separated — and the one I first blamed is the smallest

`eval/bench_retrieval.py` replays the retriever offline against the real vectors
(read-only, no Docker) and isolates each layer. On the 7 prompts whose fact provably
exists in the corpus:

| configuration | recall@8 | gain |
|---|--:|--:|
| **as-deployed** — server HNSW + semantic-only | **1/7** | — |
| exact search (i.e. fix `hnsw:search_ef`) | 4/7 | **+3** |
| + BM25 at all (shipped 0.7/0.3 weights) | 5/7 | +1 |
| + balanced fusion (0.5/0.5, min-impute) *or* RRF | 6/7 | +1 |
| remaining: `rag_ds_mac_addr34_high_offset` | — | needs ingestion work |

**(a) The vector index is configured with an almost-degenerate search beam.**
Chroma's `hnsw:search_ef` defaults to **10**; the collection sets no override
(`collection_metadata` carries only `hnsw:space=cosine`). A k=8 query with ef=10 is
barely a search at all. Concretely, on `rag_ds_ecc_curve` the true best neighbour has
cosine **0.5205** and the server's best was **0.4145** — it never visited the right
chunk. **Worth 3 of the 6 recoverable misses. This was not on my list of suspects.**

**(b) `/search` is semantic-only. It never calls the hybrid retriever.**
`llm_server.py::search_documents` → `RAGService.search()` → `collection.query()`.
No BM25, no rerank. `AdvancedRAG.hybrid_search()` is used by the **chat** path only.
And `eval/gather_rag_chunks.py` posts to `/search`. **Therefore every frozen RAG
context in `eval/results/` — the basis of every `-v2-rag` number in this repo — was
produced by a weaker retriever than the product actually uses.** Worth +1.

**(c) The fusion arithmetic — real, and the smallest of the three.**

## 1b. The fusion bug (worth 1 of 6, not the root cause)

`pipeline/advanced_rag.py::hybrid_search`:

```python
final_score = semantic_weight * result.semantic_score   # 0.7
            + keyword_weight  * result.keyword_score    # 0.3
```

Documents absent from one retriever's candidate list are **zero-filled**. A zero is
treated as *evidence of irrelevance* rather than *absence of evidence*. Consequence:

| chunk | semantic | keyword | `final_score` |
|---|--:|--:|--:|
| **best lexical hit in the whole corpus**, no semantic hit | 0.00 | 1.00 | **0.300** |
| generic chunk the embedder mildly liked | 0.45 | 0.00 | **0.315** ← wins |
| generic chunk the embedder liked | 0.57 | 0.00 | **0.399** ← wins easily |

**Break-even: any semantic-only chunk above cosine 0.4286 outranks the single best
lexical match in 6,731 chunks.** The code's own comment records that the LPUART
chunks "land at 0.42–0.57" — i.e. right at the threshold where generic chunks bury
them.

**BM25 is arithmetically incapable of winning on an identifier query.** That is why
every one of our 7 retrieval misses is an exact numeric or identifier lookup:

```
  MAC_ADDRESS34_HIGH -> offset 410        identifier + hex offset
  RSA up to 4096 / ECC up to P-521        alphanumeric spec tokens (13 and 10 chunks)
  RΘJA 22.5 °C/W                          table row
  Overdrive mode 3733 MT/s                numeric in a footnote
  Eight LPUART / Eight I2C modules        present in 53 chunks — and still missed
```

Five mechanisms in the shipped code exist to route around this and none fixes it:
`bm25_admit_hashes` (force-admit top-5 BM25), a `semantic_score > 0.3` floor,
candidate-cut preservation, a `0.6*final + 0.4*rerank` blend, and `_CONCEPT_ALIASES`
with a spliced secondary semantic search for `IOMUXC`.

**Corollary:** the `SimpleReranker` is not a cross-encoder. Its own docstring reads
*"For production, consider using a cross-encoder model like ms-marco-MiniLM."*
See §2 — following that TODO would have made things **worse**.

---

## 1c. FIXES APPLIED — measured through the real code against the live Chroma index

Verified with `eval/verify_retriever_fix.py`, which imports the actual
`RAGService` and `AdvancedRAG` and drives them against the running vectordb
container. Recall@8, per-chunk containment, 7 prompts whose fact provably exists:

| path | before | after | fix |
|---|--:|--:|---|
| `/search` (semantic) — what the eval gathered through | 1/7 | **4/7** | over-fetch |
| hybrid (chat path) | 5/7 | **6/7** | fusion min-impute |

**The `search_ef` prediction was WRONG, and this is the second time an offline
prediction got corrected by the live system today.** Setting `hnsw:search_ef=200`
via `collection.modify()` changed *nothing*: **Chroma 0.5.x ignores the metadata
`search_ef` at query time and ties the HNSW beam width to `n_results` instead.**
Proof: a datasheet chunk that is the **rank-0** nearest neighbour at `n_results=30`
is **absent from the top-8** at `n_results=8` — same query, same index. So a narrow
k=8 query cannot see its own nearest neighbour.

**Fix (a) — over-fetch, `pipeline/rag_service.py`.** `RAGService.search()` now
queries a pool of `max(k, 32)` and truncates to `k`. This is the real "+3" I had
mis-attributed to `search_ef`, delivered through `n_results`. Semantic path 1/7 → 4/7.
*No `search_ef` was added to collection creation — it would be cargo-cult; this
version of Chroma does not honour it.*

**Fix (b) — fusion min-impute + balanced weights, `pipeline/advanced_rag.py`.** A
chunk absent from one retriever's candidate list now gets that retriever's *minimum
observed* score, not zero, and weights are 0.5/0.5. Hybrid 5/7 → 6/7 (recovered
`multihop_peripheral_count`, which pure over-fetched semantic already found but the
old zero-fill fusion buried). The 0.3 semantic floor and BM25 admission were kept, so
non-RAG queries are still protected — verified: math/code/persona queries surface no
strongly-grounded chunk (top semantic 0.0), and the fictional-chip refusal probes
still find only the real chip they must refuse around.

**Fix (c) — `/search` → hybrid, `pipeline/llm_server.py`.** The endpoint (which the
eval's `gather_rag_chunks.py` posts to) now uses `hybrid_search`, matching the chat
path. So re-gathering the eval contexts will measure the product's real retriever
(6/7) instead of the semantic-only 1/7.

**Remaining miss: `rag_ds_mac_addr34_high_offset`.** BM25's tokenizer splits
`MAC_ADDRESS34_HIGH` on underscores into `mac`/`address34`/`high`, so the register
name never matches as a unit. This is the ingestion/tokenization lever the
literature flags — task #8 territory, not a fusion problem.

**Still to do:** re-gather the frozen eval contexts through the now-hybrid `/search`
(bumps the KB/eval methodology version; makes every `-v2-rag` number a product-
faithful measurement), then re-run the evals. Now unblocked — the retriever is fixed.

## 2. What the literature says (deep research, 108 agents, adversarially verified)

Full findings and citations: `docs/rag-research-findings.json`.

**Finding 1 (high confidence).** On text-and-table documents, **BM25 matches or beats
even strong dense embedders on nearly every metric**, and *no* embedder — open or
closed — reaches satisfactory first-hit recall. T2-RAGBench (EACL 2026): best
open-source embedder gets R@1 26.4; best closed gets 34.6; **none exceeds 50% MRR@5**.
The paper attributes it to tables' *"predominance of numerical values, which lack
semantic context."*
→ **Swapping the dense model will not fix this.**

**Finding 2 (high).** RRF is **not** strictly better than fixed-weight fusion — a
convex combination at **alpha = 0.5** beats tuned RRF (Recall@5 0.726 vs 0.695).
Corroborated by Bruch et al., ACM TOIS 2023.
→ **Our 0.7/0.3 is mis-weighted. Balanced is the safer default.**

**Finding 4 (high).** Replacing a lexical-heuristic reranker with a *real* neural
reranker is the largest single-component gain (+12.1pp Recall@5). **But choice is
critical: the weak, old `ms-marco-MiniLM-L6-v2` measurably *hurts* on text-and-table
data** — worse than plain BM25 hybrid. (And the +12.1pp headline model, Cohere Rerank
v4.0 Pro, is closed-source and finance-tuned; treat it as an upper bound.)

**Finding 5 (high).** Best self-hostable reranker for a 5090 is a sub-1B modern model:
**jina-reranker-v3 (0.6B, ~61.9 nDCG@10 BEIR)** or **Qwen3-Reranker-0.6B**. Both beat
`bge-reranker-v2-m3` (56.51) at equal size. *Vendor-self-reported; general BEIR, not
technical corpora.*

**Finding 6 (high).** For visually rich table/figure documents, **improving ingestion
beats swapping the embedder.** ColPali (vision late interaction, ICLR 2025) beats the
best OCR-parse-embed pipeline by **+14 nDCG@5**, and by **+14.8 on tables** specifically.
The paper's own conclusion: *"optimizing the ingestion pipeline yields much greater
performance … than optimizing the text embedding model,"* and *"little difference is
seen between BM25 and BGE-M3 embeddings."*
→ Semiconductor datasheets are exactly this document class.

**Finding 7 (high).** **BGE-M3** emits dense + sparse/lexical + ColBERT multi-vector
from one forward pass — it would replace our hand-wired MiniLM + custom BM25 + no
late-interaction with one trained encoder. *Architectural consolidation; on-domain
accuracy unproven.*

**Finding 8 (high).** **Contextual retrieval** (prepend an LLM-generated summary to
each chunk at index time) gives consistent gains on table-and-text docs. **HyDE and
multi-query expansion give little benefit for precise numeric queries** — the LLM
invents plausible-but-wrong figures and pulls the embedding off-target.

### Explicitly REFUTED during verification — do not rely on these

- "RRF is strictly better than fixed weights" — **false**.
- "RRF beats both constituents on all metrics/subsets" — **false**.
- "DAT per-query LLM weighting consistently beats fixed weights" — **not established**;
  tested only on general prose, no RRF baseline.
- "A cross-encoder reliably helps" — **false**; a weak one hurt on tables.

### The dominant caveat

**Every hard number above comes from financial text-and-table corpora or general
BEIR/MTEB — none from semiconductor datasheets.** The *mechanism* (verbatim
identifiers defeat dense embeddings) transfers cleanly and is exactly our symptom.
The *magnitudes* must be re-measured on the NXP corpus before being trusted.

And the sharpest thing the research said, which contradicted my own first instinct:

> *A gold fact present in 53 chunks yet missed at top_k=8 suggests a scoring/
> normalization bug or a broken lexical-signal path, **not** a retrieval-quality or
> coverage problem, and warrants direct debugging before any model swap.*

**It was right, and more right than I was.** I had already convinced myself the fusion
weighting was the root cause and was ready to act on it. Debugging instead of shopping
found that the fusion is worth **1 of 6** recoverable misses, while an ANN parameter I
had not suspected is worth **3**. Had I "fixed the root cause" I would have moved
recall from 1/7 to 2/7 and declared victory.

---

## 3. Plan, ordered by (measured gain ÷ cost)

**Step 0 — build the retrieval benchmark first. Nothing else is legible without it.**
We already have ground truth: `eval/repair_eval_set.py`'s `ORACLE` map names, for 7
prompts, the exact chunk containing the answer. Extend to ~100 query→gold-chunk pairs
by mining the corpus. Report **recall@k and nDCG@10**, not end-to-end answer accuracy.
*This is the entire lesson of 2026-07-09: never measure a layer through the layer above it.*

**Step 1a — set `hnsw:search_ef`. A collection metadata value. Biggest measured gain (+3/6).**
Chroma requires it at collection-creation time, so this means a re-index (or
`collection.modify(metadata=...)` where supported). Sweep ef ∈ {10, 50, 100, 200} on the
benchmark and report ANN recall vs exact. **Do this before touching a model.**

**Step 1b — make `/search` use the hybrid retriever** (or re-gather eval contexts from
the path the product actually serves). Right now the eval and the product disagree
about what "RAG" means. Worth +1/6.

**Step 1c — fix the fusion. One function, no new dependencies, no GPU.**
Balanced alpha = 0.5; **impute missing scores as the minimum of that retriever's list,
not zero**; normalize both score distributions before combining. Then **delete** the
five compensating hacks and confirm recall doesn't regress — if it does, they were
load-bearing for a reason we don't yet understand, which is itself worth knowing.
*Expected: recovers most of the 7 identifier misses. Cost: an afternoon.*

**Step 2 — replace `SimpleReranker` with a real reranker.**
`jina-reranker-v3` (0.6B) or `Qwen3-Reranker-0.6B`. **Do not use `ms-marco-MiniLM`**,
which is what our own TODO recommends and which the literature shows *hurts* on tables.
*Expected: the largest single-component gain. Cost: a model download; fits the 5090 trivially.*

**Step 3 — consolidate on BGE-M3** (dense + sparse + multi-vector from one encoder),
replacing MiniLM and the custom BM25. Only after Steps 0–2 give a baseline to compare against.

**Step 4 — fix ingestion, which the literature says is the biggest lever.**
Table-aware extraction (Docling / Marker), per-row chunking for register and spec
tables, and contextual retrieval (prepend "this row is from the i.MX 93 11×11 thermal
table" to each chunk). Evaluate ColPali-style vision late interaction — datasheets are
precisely the document class where it wins by +14.8 nDCG@5 on tables.

**Step 5 — only then ingest the 265 new NXP PDFs** (~1.1 GiB, verified to contain
`Neutron`, i.MX 95 core counts, GPIO counts). Adding 4× the corpus to a 75%-recall
retriever lowers recall and confounds every eval. **Blocked on Steps 0–1.**

---

## 4. What this changes about our published numbers

Every RAG pass rate in this repo — including production Skippy's 73.8% / 63.5% — is a
**joint measurement of the model and a retriever with 25% miss rate on answerable
datasheet questions.** A meaningful fraction of what we reported as model error was
retrieval error. The two are indistinguishable in an end-to-end substring score.

Not a reason to discard the numbers. A reason to stop calling them model numbers.
