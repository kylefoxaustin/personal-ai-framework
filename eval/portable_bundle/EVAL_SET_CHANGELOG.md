# Eval set changelog

Changes to `eval/prompts_v2.json` over time. Every change here also bumps
`methodology_version` in the data bundle.

---

## 2026-05-08 — SK-P0-001: persona category quarantined

**Trigger.** External Claude review (`REMEDIATION_PLAN.md`) flagged that the
`persona` category scores 0/6 for every model in the bundle (stock Qwen 7B,
stock Qwen 32B, stock Mistral 7B, stock Qwen3-30B-A3B, every Skippy fine-tune
including production). A constant −6 contribution to every headline that
doesn't differentiate models.

**Diagnosis.** Persona is structurally incompatible with substring grading.
Both persona prompts (`persona_skippy_voice`, `persona_brief_role`) shipped
with empty `gold_substrings`, which causes the grader to return `"manual"`
status — not a pass. Even if we added gold substrings (e.g., `["Skippy"]`),
the deployment system prompt injects "Skippy" identity into every model
including stock bases, so the substring doesn't differentiate FT persona
transfer from default-system-prompt behavior. Persona/voice transfer is
real, but it's measured by `eval/voice_metrics.py` (length, bullets, bolds,
emojis, opener-boilerplate), not by substring matching.

**Action.**

1. Added `category_status: "BROKEN_SUBSTRING_INCOMPATIBLE"` field to both
   persona prompts in `prompts_v2.json` with an explanatory note.
2. Wrote `eval/regrade_for_broken_categories.py` to retroactively recompute
   `summary.passed`, `summary.total`, and `summary.pass_rate` in every
   existing eval JSON, excluding broken prompts. Original numbers preserved
   in a new `summary_v1_legacy` field for audit trail.
3. New denominator: 42 prompts × 3 samples = **126 samples** (was 132).
4. Headline shifts: most v4-era runs gain +3pp (persona was 0/6 dead weight).
   Per-category Δ between models is unchanged in absolute pass-count terms;
   percentage Δs shift slightly because the denominator changed.

**Impact on stated conclusions.**

Direction of every load-bearing finding is preserved:
- Skippy 7B v4 (production): 70.5% → **73.8%**, still passes capability gate
- 14B v4 fabricates: unchanged (persona unrelated)
- Mistral v4 vs stock Mistral: was −3.8pp, now **−4.0pp** (regression slightly
  larger after persona removal — gotcha #7 finding direction holds)
- 32B v4 vs 32B base: was −4.6pp, now **−4.7pp** (corpus-too-small finding holds)
- MoE attention-only catastrophic regression: was −9.8pp, now **−10.3pp** (holds)
- MoE +router recovers: was −3.8pp from base, now **−4.0pp** (holds)

**Files affected.**

- `eval/prompts_v2.json` — persona prompts now flagged
- `eval/results/acc_*.json` — every eval JSON's `summary` block updated;
  `summary_v1_legacy` preserves original numbers
- `eval/regrade_for_broken_categories.py` — new, re-runnable
- `docs/skippy-data-bundle.xlsx` — re-built from updated JSONs
- `docs/skippy-white-paper.md` — pending updates to headline numbers
- `docs/recipe-taxonomy.md` — pending updates to filled-cell headlines
- `personal-ai-use-cases.pptx` — pending deck regen for new numbers

**methodology_version bumped:** `2026-05-08-post-remediation`

---

## 2026-07-09 — the quarantine was never applied to the producer (or to subdirs)

**Trigger.** While semantically regrading the W8A8 INT8 pair (to test whether the
headline −3.8pp was a substring-grading artifact), the regrade tool reported a
suspicious "+3.74pp semantic gain" on both arms. It wasn't a gain.

**Two bugs, both dating from the 2026-05-08 remediation.**

1. **`regrade_for_broken_categories.py` used a non-recursive `glob`.**
   `RESULTS.glob("acc_*.json")` never descended into `eval/results/runpod/` or
   `eval/results/runpod-int8/` — i.e. it missed **exactly the FP8 and INT8
   quantization runs**, the data the entire precision-ladder / NPU-silicon
   argument rests on. Those 6 files still carried the pre-quarantine denominator.

2. **The 2026-05-08 fix was applied to data, not to the producer.**
   `run_accuracy_eval.py` still emitted `total_samples: 132` with no exclusion, so
   *every eval run after that date silently re-introduced the old denominator* —
   the 26-run variance suite (SK-P0-002) and the entire Gemma/Yi/Phi-4/Llama
   cross-family v4 campaign of 2026-05-09/10. Net state before this fix:
   **36 result files on n=126, 41 on n=132.** Anyone comparing across those two
   groups was mixing denominators.

**A third bug fell out of the first.** `regrade_semantic.py` skipped persona by
checking the `category_status` **flag**, which pre-2026-05-08 result JSONs don't
carry (they embed their own older prompt copies). So on those files it *graded*
persona — which the LLM judge passes 6/6 and the substring grader scores 0/6 —
and counted it into `summary_semantic`. That inflated every semantic-vs-substring
Δ on the affected files by ~4.5pp, making semantic grading look uniformly
generous. **It is not.** Persona-excluded, semantic is *stricter* than substring,
and the sign of the gap is model-family-dependent:

| model | substring | semantic | Δ |
|---|--:|--:|--:|
| Mistral-7B-Instruct | 63.5% | 65.9% | +2.4 |
| Gemma-2-9B-it | 61.9% | 65.1% | +3.2 |
| Llama-3.1-8B-Instruct | 59.5% | 61.1% | +1.6 |
| **Skippy 7B v4 (Qwen)** | **73.8%** | **63.5%** | **−10.3** |

That is Finding 4 (the Qwen-family substring bias) in a single table.

**Action.**

1. `regrade_for_broken_categories.py` → `rglob`, and skip `*_semantic.json`.
2. `regrade_semantic.py` → skip by prompt **ID** as well as flag; added
   `BROKEN_PROMPT_IDS`, kept in sync with the other two scripts.
3. **`run_accuracy_eval.py` → excludes persona at the source.** New runs emit
   `summary.total = 126`, `excluded_samples = 6`. Persona is still *run and
   recorded* (`eval/voice_metrics.py` grades voice transfer from those samples);
   it is simply not in the denominator. `summary.total_samples` is retained as a
   legacy field only — **never divide `passed` by it.**
4. Re-ran the remediation across all 77 result JSONs. `summary_v1_legacy`
   preserved everywhere. Backup: `eval/results_backup_20260709-185111.tar.gz`.
5. Recomputed `summary_semantic` persona-excluded in all 35 `*_semantic.json`.
   Only the 2 RunPod INT8 files changed materially (132 → 126).

**Impact on stated conclusions.**

The W8A8 result is *strengthened*, not weakened. Persona-excluded, n=126:

| grader | fp16 | INT8 W8A8 | Δ |
|---|--:|--:|--:|
| substring | 86/126 = 68.2% | 81/126 = 64.3% | −3.96 pp |
| semantic (GPT-4o) | 85/126 = 67.5% | 80/126 = 63.5% | −3.97 pp |

**The two graders agree to 0.01pp** — the INT8 penalty is not a grading artifact.
Per-category (semantic), *all* loss is in retrieval-grounded generation:
`rag_datasheet` −3, `multihop` −1, `general` −1; **`refusal` 9/9 → 9/9, zero
loss**, retiring the old "refusal specificity" framing for good. `coding`,
`reasoning`, and `numerical_precision` are untouched.

**methodology_version bumped:** `2026-07-09-persona-exclusion-fix`

---

## 2026-07-09 — v2.1: the eval set was measuring the retriever, not the model

**Trigger.** A "−3.97pp INT8 capability loss" (published to three sessions) was
traced to the eval set. See the entry above for the grading half of the story;
this is the data half.

**Audit method.** Every prompt's `gold_substrings` were checked against (a) its own
frozen RAG context and (b) the actual ChromaDB corpus (`vectordb/chroma.sqlite3`,
6,731 chunks) using a **proximity test** — the gold token must appear within ~140
chars of a topical co-token. A naive substring probe was itself fooled: `'3733'`
matched email folder paths (`Inbox/3733`), `'148'` matched the page number
`1480 / 5652`, and `'149'` matched the application note `AN14149`.

**Three defect classes found (18 of 44 prompts):**

1. **7 prompts — retriever miss.** The gold fact IS in the corpus; the k=8 hybrid
   retriever never surfaced it. `RSA up to 4096, ECC curves up to P-521` sits in 13
   and 10 chunks. `Eight Low Power I2C modules … Eight LPUART modules` in 53.
   `Overdrive mode 3733 MT/s` in 26. `RΘJA 22.5 °C/W` in 3. **These were scored as
   model failures. They were retrieval failures.**

2. **6 prompts — the fact is not in the knowledge base at all.** Including two
   with *contradictory* golds (i.MX 93 GPIO count asserted as both 148 and 149;
   neither appears anywhere). `'Neutron'` appears in **0 of 6731 chunks**.
   A perfect RAG model scores zero on all six.

3. **4 prompts — weak gold.** Every gold token so common in its own context that
   any fluent answer contains it (`'i.MX 93'` ×100, `'LPDDR4'` ×49).

**Repairs** (`eval/repair_eval_set.py`, every change emitted with its evidence):

- **Class 1 → oracle chunks.** The chunk containing the fact is spliced in at rank
  1, lowest-ranked chunk dropped, k held at 8. Labelled `retrieval: "oracle"`.
  This is what the corpus already *claimed* to do — freeze retrieval so any
  difference is attributable to the model, never the retriever. **It measures
  reading comprehension, not end-to-end RAG. Do not conflate the two.**
- **Class 2 → hallucination probes.** `category: "faithfulness"`,
  `match_mode: "any"`, gold = refusal phrasings, plus `unanswerable_from_context`
  and an `evidence` field. Refusing is now correct; asserting the fact is
  parametric bleed-through, which for a RAG assistant is the failure mode of
  interest. These are **harder than the existing `refusal` prompts because the
  chips are real.**
- **Class 3 → tightened golds.** `['A55','M33']` → `['Cortex-A55','Cortex-M33']`;
  `['1.7','GHz']` → `['1.7 GHz']`; `['LPDDR4']` → `['3733','MT/s']` (the question
  asks for the *rate*, not the DRAM type).
- `general_embedded_book` (empty gold, a book recommendation) quarantined like
  `persona`.

**Result.** `eval/portable_bundle/check_eval_integrity.py` exits 0 on v2.1.

| | v2.0 | v2.1 |
|---|--:|--:|
| prompts | 44 | 44 |
| quarantined | 2 | 3 |
| **scored samples** | **126** | **123** |
| `faithfulness` | 0 | **6** |
| `rag_datasheet` | 26 | 23 |
| `multihop` | 3 | 1 |
| `rag_email` | 1 | 0 |

**Validation.** A model that always answers *"I don't have that information"* now
passes **exactly** the 9 refusal + faithfulness prompts (27/123) and nothing else.

**⚠️ Comparability.** v2.1 numbers are **not** comparable to v2.0 numbers on the 17
touched prompts. Historical results remain valid *as v2.0 measurements*. Do not
mix. `methodology_version: 2026-07-09-eval-set-v2.1-repaired`.

**Still open:** the retriever missed facts sitting in 13, 26, 53 chunks at k=8.
That is a **retriever bug, and it is a bigger finding than the eval bug** — it
means production Skippy is also failing to surface these facts. Not yet
investigated.
