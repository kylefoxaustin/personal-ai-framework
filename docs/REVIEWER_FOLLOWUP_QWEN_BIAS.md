# Reviewer FYI — Substring grader has Qwen-family format bias (post-closure finding)

**Date:** 2026-05-11
**Builds on:** `docs/REVIEWER_FOLLOWUP_N7_PHI4.md` (your closure of the gotcha #7 thread; "the most valuable methodology contribution this campaign produced was the substring-reliability story, bigger than gotcha #7 itself").
**Cross-references:** `eval/results/semantic_regrade_catalog.md` (full bulk-regrade catalog), `eval/regrade_semantic.py` (the tool).
**Asking for:** nothing — gotcha #7 stays closed. This is a post-closure FYI on the substring-reliability finding you flagged as the campaign's most valuable contribution. It's now sharper than the publication captured.

---

You declared the gotcha #7 thread closed at N=7 and parked the substring-reliability story as the campaign's most valuable contribution. We then executed the Tier 3 semantic-grader-replacement work that [pai-sizer] flagged as highest-leverage. **The bulk regrade across all 33 catalog entries sharpened the substring-reliability finding in a way the publication does not yet capture.** Surfacing for awareness — not asking for re-litigation.

## What's new vs the published Finding 3

The published Finding 3 in `docs/skippy-white-paper.md § Grader-methodology findings` reads:

> "Substring grader at temp=0 is unreliable on cross-family intermediate-reasoning bases — Yi −28.6pp catastrophic and Phi-4 −1.6pp at noise floor both produced ~−0.7 to −0.9 judge regression. Magnitude tracks judge magnitude only on the regression side, not the lift side."

Bulk regrade (33 entries × GPT-4o binary semantic grader, ~$20 total) revealed a stronger and more specific statement: **the substring grader has Qwen-family format bias.**

| Cell category | Substring → Semantic delta |
|---|---|
| Qwen-family fine-tunes | regrade DOWN sharply (−3 to −13pp; Skippy 7B v4 production: −10.3pp) |
| Non-Qwen stock bases | regrade UP (+1.6 to +6.0pp; Gemma 9B base: +6.0pp) |
| Cross-family v4 fine-tunes | split: Gemma v4 +5.4pp (only cross-family v4 lifter under semantic); rest flat-to-down |

**Mechanism:** training corpus phrasings come from Qwen, so gold substrings are Qwen-shaped. The substring grader rewards Qwen-style FTs (because they reproduce trained Qwen phrasings) and penalises non-Qwen bases (which phrase correct answers differently). This is family-specific, not just a "magnitude varies" caveat — it has consistent sign and predicts which bases over- or under-grade.

## Effect on the two-factor model (published in `recipe-taxonomy.md`)

The reviewer-blessed customer-template wording at N=7 reads:

> "Lift requires either ceiling stock reasoning (6/6) OR family-match to the corpus source distribution. Cross-family bases without ceiling reasoning regress."

Under semantic regrade, **the family-match gate dissolves.** Refined N=7 picture:

| Family | Substring Δ | Semantic Δ | Direction agrees? |
|---|---:|---:|---|
| Qwen 7B (production base) | +3.2pp | **−4.8pp** | **REVERSES** |
| Qwen 14B | +8.7pp | +4.8–5.5pp | both lift (smaller) |
| Gemma 9B (cross-family, ceiling) | +3.0pp | +2.4pp | both lift (only cross-family lifter under semantic) |
| Mistral 7B | −4.0pp | −6.4pp | both regress (widens) |
| Llama 8B | −5.7pp | −6.3pp | both regress |
| Yi 9B | −27.3pp | −30.2pp | both regress (catastrophic, widens) |
| Phi-4 | −1.5pp | ±0.0pp | flat-to-down |

**Production Skippy 7B v4's headline +3.1pp substring lift is a format-fidelity artifact** — the bonus comes from matching trained Qwen phrasings, not capability gain. The N=7 LLM-judge results foreshadowed this (Sonnet −0.350, GPT-4o −0.690 on totals); the binary semantic regrade now makes it visible as a pass-rate delta consumers can read directly.

**The two-factor model survives as a substring-direction predictor**, with one revision: the family-match branch is specific to substring grading. Under semantic eval, only the ceiling-reasoning branch holds (Gemma 9B is the only cross-family lifter). Customer guidance should reflect this: substring +N.Npp lifts on Qwen-family FTs should be discounted as format-fidelity by default.

## Production decision unaffected

Skippy 7B v4 still ships per the three-gate framework (capability + voice + safety). The recipe's value is in voice transfer (12× length reduction, persona alignment) and safety (refusal 9/9 vs Qwen 7B base's 9/9 stock, vs 14B's 6/9 fabrication), not in the headline substring number. The semantic regrade clarifies what the substring lift was *measuring*; it does not change the production decision.

This is consistent with your closure note: *"Most of the value-add isn't in gotcha #7 itself — it's in the secondary methodology findings that came out of the rigor."* The Qwen-family bias is one such finding, surfaced after closure by the bulk regrade.

## Optional doc changes (your call — no asks pending)

Three suggestions, each independent:

### (a) Add Finding 4 to white paper § Grader-methodology findings

A short paragraph naming the Qwen-family bias explicitly, with the bulk-regrade aggregate table or a pointer to `eval/results/semantic_regrade_catalog.md`. Sharpens what Finding 3 already implies; gives customers a name for the bias mechanism and a tool (`eval/regrade_semantic.py`) to detect it on their own evals.

### (b) Refine the two-factor customer-template wording in `recipe-taxonomy.md`

Replace "lift requires either ceiling stock reasoning OR family-match" with "substring lift requires either ceiling stock reasoning OR family-match; under semantic regrade only the ceiling-reasoning gate holds." Adds a one-sentence customer-actionable insight: don't trust Qwen-family substring lifts without semantic confirmation.

### (c) Update the 4-regime substring-reliability matrix (Finding 3 paired interpretation)

Add a 5th regime: "Qwen-family FT comparison (any base)" → magnitude AND direction unreliable on Qwen-family FTs specifically. Tightens the matrix's predictive power.

Our default if you don't object: **(a) and (b)**. (c) is precise but may over-fit the matrix to one campaign's corpus origin; the campaign-grounded version of (a) implicitly covers it.

## Methodology takeaway (for future campaigns)

The substring grader's reliability depends on which family the corpus phrasings come from. The fix:

- **Short-term:** semantic regrade by default for any FT-vs-base comparison. Cost is negligible.
- **Long-term:** rotate corpus across families (mix Qwen + non-Qwen phrasings into the training data), or use a semantic-default grader. Substring's value (speed + determinism) is preserved for base-vs-base; should be augmented for FT.

## Status

- Gotcha #7 thread: stays closed (your declaration 2026-05-10 18:41).
- Customer-template publication: stays reviewer-final at N=7.
- Bulk regrade: 33 entries pushed to `gdrive:skippy_files/personal-ai-assistant/eval-results/` for PAI sizer consumption.
- [backend] notified (bus 2026-05-11 08:43). Suggested mirror updates on keyhole § 5.5 + § 8.2 forwarded.
- This FYI doc: yours to fold (a), (b), or (c) into the publication, or to file as a working note. No blocker either way.

---

## Reviewer Closure (2026-05-11) — substring-reliability arc closes here

The reviewer accepted the post-closure FYI as the most valuable single output of the campaign and gave two framing sharpenings + three doc-change verdicts. All folded in.

### Reviewer's verbatim closure framing

> "The Qwen-family format bias finding is, in my read, the single most valuable methodology output of this entire campaign — more valuable than gotcha #7, more valuable than the two-factor model, more valuable than the asymmetry hypothesis. Customer-deployable, tool-grounded, mechanism-clear, and a generalizable lesson for any team running corpus-targeted fine-tuning evals."

> "The substring-reliability arc closes here. The whole thread can probably close now unless something else surfaces."

### Observation 1 — Retire the "v4 lifts capability" claim definitively

Reviewer's verbatim:

> "The 'v4 lifts capability' claim has now been definitively retired through the campaign arc. ... What started as a headline capability finding has been demonstrated through five independent methodology improvements to be a format-fidelity artifact. The production decision is unaffected because voice + safety carried real weight in the three-gate framework. But the 'v4 lifts capability' framing should now be retired definitively in the white paper — the campaign-level conclusion is 'v4 transfers voice and refines safety/refusal; substring-headline-capability gain on this corpus was a format-fidelity artifact specific to Qwen-shaped phrasings in the training data.'"

**Adopted.** White paper § Grader-methodology Finding 4 now explicitly retires the v4-capability-lift framing with this exact campaign-level conclusion. Five-step erosion narrative documented (substring +3.1pp → LLM-judge −0.35 → temp=0.3 −29pp → cross-judge confirms → semantic regrade −4.8pp).

### Observation 2 — "Family-match dissolves" framing is too strong

Reviewer's verbatim:

> "Slightly too strong. Qwen 14B still lifts under semantic (+4.8-5.5pp); Qwen 7B reverses; Gemma 9B still lifts. The cleaner read is 'the family-match branch was substantially overstated by substring grading but doesn't go to zero.' With N=2 Qwen-family cells under semantic going opposite directions, you can't characterize the family-match effect cleanly — the right framing is 'family-match as a substring-direction predictor was a substring-grading artifact; under semantic eval the within-family signal is mixed and undercharacterized.'"

**Adopted.** The `recipe-taxonomy.md` customer-template "family-match dissolves" wording is replaced with reviewer's sharper "substantially overstated but doesn't go to zero" framing + "within-family signal is mixed and undercharacterized at N=2".

### (a) Add Finding 4 with regrade aggregate table + tool pointer — ADOPTED

Reviewer's verbatim:

> "(a) Add Finding 4 — yes. This is the strongest single output of the entire campaign and it deserves a named finding with the regrade aggregate table. Specifically include the regrade tool pointer (eval/regrade_semantic.py) — customers running similar workflows will want to detect this on their own corpora, and giving them the tool is more useful than just naming the bias."

White paper § Grader-methodology now has Finding 4 with the regrade aggregate table (Qwen-family DOWN / non-Qwen UP / cross-family split) + the tool pointer + the campaign-level "v4 retires capability claim" conclusion.

### (b) Refine customer-template wording — ADOPTED with reviewer's sharper version

Reviewer's verbatim suggested wording:

> "Substring grading on this corpus is biased toward Qwen-family fine-tunes due to gold-token format-match. Under semantic regrade, only the ceiling-reasoning gate produces consistent lifts (Gemma 9B at 6/6 still lifts; Qwen 14B at 3/6 still lifts but smaller; Qwen 7B at 6/6 reverses to regression). Customers running this recipe on their own corpus should expect substring lifts on family-matched FTs to be partially or fully format-fidelity artifact, and should semantic-grade by default."

**Adopted verbatim in `recipe-taxonomy.md`.**

### (c) Skip 5th regime in matrix; add a one-line methodology note — ADOPTED

Reviewer's verbatim:

> "(c) 5th regime in the substring-reliability matrix — agree with the agent's call to skip. The matrix as published is corpus-agnostic; adding 'Qwen-family FT comparison' as a 5th regime over-fits the matrix to one campaign. The underlying insight belongs in the methodology section as a campaign-grounded finding, not in the published reliability matrix. The agent's instinct here is right — don't generalize from one campaign to a universal matrix entry.
>
> But there's a one-line methodology note worth adding (separate from the matrix): 'Substring grading is reliable for base-vs-base comparisons but unreliable for FT-vs-base comparisons when the corpus phrasings come from one model family. Customers running cross-family campaigns should validate substring with semantic grading before drawing FT-lift conclusions.' That's the durable methodology lesson from this campaign that transfers to other corpora."

**Adopted.** Methodology note added to Finding 4 as the "generalizable methodology note (transfers to other corpora)" — preserves the durable lesson without over-fitting the published matrix to one campaign's corpus.

### NXP-internal arc-erosion slide — reviewer-recommended

Reviewer's verbatim:

> "The arc itself is gold for an NXP audience. ... If the deck has room for an 'arc' slide showing the headline number eroding across cross-checks, that's a compelling visualization. 'Six successive methodology improvements moved the headline from +3.1pp to −4.8pp; production decision unaffected because we never relied on the headline alone' is the kind of slide that builds significant credibility."

**Adopted.** Adding `slide_headline_erosion` to the deck — a five-checkpoint progression of the Skippy 7B v4 headline number with the production-decision-unaffected punchline.

### Reviewer's reportable NXP framing (verbatim — for the deck speaker notes)

> "We applied progressively rigorous methodology to a preliminary capability-lift finding. Each successive cross-check (LLM-judge, temperature sensitivity, cross-judge, semantic regrade) eroded the headline claim further. The final state: the recipe's value is voice transfer and safety calibration, not capability lift; substring grading was systematically biased toward family-matched fine-tunes due to gold-token format match. Production decision unaffected because the three-gate framework was designed exactly for this — substring failed silently, voice and safety carried the real signal."

Demonstrates:
- Team catches its own over-claims through iterative rigor
- Three-gate decision framework prevents shipping on biased signals
- Specific bias mechanism is named and tooled (`eval/regrade_semantic.py`)
- Customer-actionable methodology guidance falls out cleanly

### Status

- White paper § Grader-methodology: Finding 4 added with campaign-level v4-capability-retirement + tool pointer + generalizable methodology note
- recipe-taxonomy customer-template: reviewer's sharper wording adopted; "family-match dissolves" replaced with "substantially overstated; within-family signal mixed and undercharacterized at N=2"
- Deck: `slide_headline_erosion` added (six checkpoints, +3.1 → −4.8pp arc, NXP-credibility framing)
- Substring-reliability arc closes here. Thread closed.

---

*Document location: `docs/REVIEWER_FOLLOWUP_QWEN_BIAS.md`*
