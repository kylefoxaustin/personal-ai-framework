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
