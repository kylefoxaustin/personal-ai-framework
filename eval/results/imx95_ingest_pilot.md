# i.MX 95 ingestion pilot — measured

**2026-07-11.** First slice of the "265 un-ingested NXP PDFs" task, run as a
*measured* pilot rather than a bulk ingest, because the corpus was i.MX 93-heavy
and the retriever thread flagged cross-chip contamination as a real risk.

## What was ingested

| file | pages | chunks | note |
|---|--:|--:|---|
| `IMX95RM.pdf` | 11,263 | **9,070** | i.MX 95 Reference Manual — bigger than the entire prior corpus |
| `IMX95IEC.pdf` | 125 | 112 | industrial datasheet |
| `AN14120.pdf` | 19 | 14 | app note (detected as i.MX 8M) |

Corpus: **6,731 → 15,927 chunks (2.37×).** i.MX 95 is now the single largest
chip in the store (9,182 `part_number` chunks vs i.MX 93's ~5,800).

## Test 1 — did it degrade i.MX 93 retrieval? (the contamination risk)

Instrument: `eval/bench_retrieval.py` (offline, read-only against
`chroma.sqlite3`; encoder self-check cosine = 1.0000 both runs). Same 7 i.MX 93
gold prompts, before vs after. **Production retriever = hybrid 0.7/0.3 zero-fill.**

| fusion | R@8 before | R@8 after |
|---|:--:|:--:|
| semantic-only | 4/7 | 4/7 |
| **hybrid 0.7/0.3 (SHIPPED)** | **5/7** | **5/7** |
| hybrid 0.5/0.5 min-impute (stashed) | 6/7 | 5/7 ⚠️ |
| RRF k=60 | 6/7 | 6/7 |

**The shipped retriever held exactly — 5/7, same two misses (`mac_addr34_high_offset`,
`imx93_uart`).** No production regression, despite i.MX 95 now outweighing i.MX 93.
The only casualty was the *experimental* min-impute arm (`imx93_uart` displaced from
top-8); RRF held it. Baseline preserved at `retrieval_bench_BASELINE_pre_imx95.json`.

## Test 2 — did it add i.MX 95 coverage? (the point of ingesting)

Live `/search` (hybrid), queries unanswerable before this ingest:

- ✅ `i.MX 95 Neutron NPU TOPS` → `IMX95RM.pdf` **Ch. 177 NPU Domain Overview**
- ✅ `i.MX 95 eMMC boot` → `IMX95RM.pdf` boot-partition / BOOT_MODE sections
- ⚠️ `i.MX 95 Cortex-A55 application cores` → returned **i.MX 93** IEC/CEC/XEC tables

## The one real finding: contamination is *query-dependent*

Chip-**specific** vocabulary (Neutron, eMMC BOOT_MODE) routes correctly to i.MX 95.
Chip-**shared** boilerplate ("Cortex-A55 core count") can cross-contaminate: the
i.MX 93 datasheet's dense "Page 2 Tables" chunks outrank the new chip's prose. So
the hazard is not "i.MX 95 hurts i.MX 93" (it didn't) — it's "on shared-architecture
terms, the denser-tabled incumbent chip outranks the newcomer."

**Fix is already latent in the data:** every chunk carries a `part_number`
metadata tag (populated: I.MX 95 = 9,182, I.MX 93 = 5,356, …). A retrieval-time
chip filter/boost — when the query names a chip, restrict or up-weight that
`part_number` — closes the cross-chip case without touching fusion. This is exactly
the "reduce over-grounding / wrong-chip" headroom the retriever thread named.

## Verdict

**Green light to expand**, with two conditions:
1. Keep measuring per-slice (RT1180 = 117 files, MCXN947 = 28, i.MX 91 = 13 will
   pile far more shared-boilerplate pressure than one i.MX 95 RM did).
2. Build the `part_number` chip-scoped retrieval filter **before** the big RT1180
   dump — that's when shared-vocab collisions (all these are Arm SoCs) get dense.

---

# Full expansion COMPLETE (2026-07-11)

Chip filter built (`advanced_rag.py::hybrid_search` step 5b + `_chip_key`), then the
remaining ~250 PDFs ingested: **RT1180 (89), MCXN947 (28), i.MX 91 (16)** = 133 files
processed, 16,352 chunks. Corpus **6,731 → 31,904 (4.74×)**, five chips.

## Contamination test at full scale — the "4× makes it worse" fear is REFUTED

Same 7 i.MX 93 gold prompts, `bench_retrieval.py`, production hybrid 0.7/0.3, recall@8:

| corpus | chunks | R@8 |
|---|--:|:--:|
| baseline (i.MX 93 only) | 6,731 | 5/7 |
| + i.MX 95 | 15,927 | 5/7 |
| **+ RT1180 + MCXN947 + i.MX 91 (all 5 chips)** | **31,904** | **6/7** |

Production i.MX 93 recall **held and marginally improved** across a 4.74× corpus
(`mac_addr34` flipped to HIT on a BM25-IDF shift from the larger corpus; `uart` is the
one persistent miss at every size). Adding four chips did not degrade the incumbent.

## Chip filter routes all five correctly (live `/search`)

Each chip-named query lands on its own chip's docs — no cross-chip contamination:
`RT1180 ENET`→IMXRT1180RM · `MCXN947 PowerQuad`→MCXNP184…RM · `i.MX 91 A55`→IMX91RM ·
`i.MX 93 UART`→IMX93RM · `i.MX 95 Neutron`→IMX95RM.
(Minor: MCXN `part_number` extraction is weaker than the i.MX pattern — the MCXN part
is a `MCXNP184…` variant, tagged chip-unknown rather than `mcxn947`, so it's neutral
not boosted; it still wins on relevance. Fine in practice; tighten if MCXN grows.)

## Housekeeping
- Ingester `clean_text` had a benign embedded `\x01` in its repeated-char regex (a
  no-op after control-char stripping); left as-is. The MCXN RM "hang" was just a slow
  3,763-page extraction, not a bug — misdiagnosed then measured.
- Test `_probe.txt` chunk removed (deleted from inside the llm-server container; the
  local chromadb 1.5.5 client can't talk to the 0.5.0 server).
- `part_number` extractor extended with RT11xx/MIMXRT/MCXN patterns so new chips tag.
