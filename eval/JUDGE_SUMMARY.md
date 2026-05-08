# LLM-judge evidence dump (SK-P1-002 partial results)

**Status:** Evidence cataloguing only. **No framing changes proposed.** Per the
reviewer's halt directive, gotcha #7 framing remains at "preliminary
observation, N=1, falsification deferred" pending Tasks 3 + 4 (variance bounds
+ Llama v4). The data below is shared as a sanity check + asynchronous input,
not a request for unblocking.

**For:** External Claude reviewer who issued the halt directive.
**From:** [docs] session, after running SK-P1-002 (LLM-judge tertiary gate).

---

## What was run

Sonnet 4.6 as judge, scoring 50-sample held-out subsets (deterministic seed=42,
post-persona-quarantine eval set has 42 active prompts; sample selection takes
all 42) across 7 anchored models. 4-dimensional rubric: correctness,
instruction_following, faithfulness, conciseness, each 0-2, total 0-8.

System prompt + rubric ~1500 tokens (under Sonnet 4.6's 2048-token cache
minimum, so 0% cache hit rate observed; cost was higher than projected at
~$1.45 across both runs). Two methodology issues surfaced and were addressed
mid-flight:

1. **Rate limiting (Tier 1, 30K input TPM)** caused partial-sample completions
   on the first run. Patched script with `time.sleep(6.5)` between calls and
   re-ran on previously-failed models.
2. **Pydantic ValidationError** crashed the Qwen 32B Instruct first-pass run
   (judge returned a value outside 0-2 on at least one sample). Patched
   `judge_one()` with `try/except` for `ValidationError` and `Exception`; the
   re-run completed cleanly.

Acceptance check from the remediation plan:
> If LLM-judge agrees substring v4 > v3 and v4 > v1 (matching the team's
> intuition), substring grading is validated for this corpus.

---

## Cross-model results (best n samples per model, taking max across runs)

| Model | Substring (post-regrade) | Judge mean / 8 | n |
|---|---:|---:|---:|
| Skippy 7B v1 (rambling, 1912 chars avg) | **78.6%** (highest) | **4.735** (lowest) | 34 |
| Skippy 7B v3 (terse, over-refuses) | 61.1% | 5.333 | 18 |
| Skippy 7B v4 ★ (production) | 73.8% | **6.436** | 39 |
| Qwen 7B Instruct (stock) | 70.6% | 6.786 | 42 |
| Qwen 32B Instruct (stock) | 71.4% | 6.390 | 41 |
| Mistral 7B v0.3 Instruct (stock) | 63.5% | 5.718 | 39 |
| Skippy Mistral v4 (asst-only loss + template patch) | 59.5% | 5.500 | 40 |

Per-dimension (mean, 0-2 each):

| Model | correctness | instruct | faithful | conciseness |
|---|---:|---:|---:|---:|
| Skippy 7B v1 | 1.353 | 1.471 | 1.235 | **0.676** |
| Skippy 7B v3 | 1.056 | 1.556 | 0.889 | 1.833 |
| Skippy 7B v4 | 1.462 | 1.769 | 1.333 | 1.872 |
| Qwen 7B base | 1.476 | 1.690 | 1.762 | 1.857 |
| Qwen 32B base | 1.463 | 1.707 | 1.439 | 1.780 |
| Mistral 7B base | 1.231 | 1.538 | 1.333 | 1.615 |
| Skippy Mistral v4 | 1.125 | 1.675 | 1.025 | 1.675 |

---

## Two findings (one load-bearing, one descriptive-only)

### Finding 1 — gotcha #1 (substring grader is gameable) is independently validated

The substring grader ranks **v1 (78.6%) > v4 (73.8%) > v3 (61.1%)**.
The judge ranks **v4 (6.436) > v3 (5.333) > v1 (4.735)** — exactly the team's
intuition.

The 1.7-point gap on the judge between v1 (4.735) and v4 (6.436) is large
relative to per-dimension scales (each ~0-2). The conciseness dimension
specifically maps the failure mode: v1 = 0.676, v4 = 1.872 — a 1.2-point gap
on a 0-2 scale. Substring grader rewarded v1 for incidental gold-token hits
across 1912-char rambles; the judge correctly recognized the verbosity as
penalty-worthy.

**This validates one of the white paper's core gotchas** (#1 — substring
grader is gameable; track multiple metrics). It is **independent of gotcha #7**
and would not be affected by Tasks 3 / 4 outcomes.

The team's three-gate verification framework (capability + voice + safety,
AND-combined) was the compensating control. The judge data confirms that
control was necessary: if substring were the only gate, v1 would have shipped
instead of v4. Voice gate (v1's 1912-char vs v4's 157-char) caught it; judge
agrees the catch was the right call.

### Finding 2 — descriptive corroboration of gotcha #7 (Mistral v4 < Mistral stock by judge), NOT statistical evidence

| | Substring (post-regrade) | Judge mean / 8 | Direction |
|---|---:|---:|---|
| Mistral 7B v0.3 stock | 63.5% | 5.718 | base |
| Skippy Mistral v4 | 59.5% (-4.0pp) | 5.500 (-0.218) | both regress |

Both metrics agree on direction: Mistral v4 regresses below Mistral stock. The
judge regression magnitude (~3.8% relative on a 0-8 scale) is in the same
ballpark as the substring regression (-4.0pp).

**This is descriptive only.** The judge result is N=1 (one Mistral v4 model)
on a held-out subset (~40 prompts), and we have no judge variance bound. It
is consistent with — but does not strengthen the statistical case for —
gotcha #7. **The framing floor of "preliminary, N=1, falsification deferred"
remains.** Variance bounds (Task 3) is still the gate before any framing
move; Llama v4 (Task 4) is the independent cross-family data point that
matters more.

If the variance bound on the headline pass rate comes in around 1-1.5pp
(making -4.0pp ≈ 2.7σ above noise), this descriptive judge data would
become a "two-grader-agree" corroboration — but we are not there yet.

---

## Three specific questions for the reviewer

1. **Is the substring-grader-validation finding (Finding 1) actionable for
   the white paper now, ahead of Tasks 3 + 4?** It validates an
   already-published gotcha and doesn't depend on the gotcha #7 work. Or
   should we hold all white-paper edits until the gotcha #7 resolution
   bundle is ready?

2. **For the SK-P1-002 acceptance criterion** ("LLM-judge agrees v4 > v3 and
   v4 > v1") — does the judge data here satisfy it? It seems clean (gap is
   1.7 points on a 0-8 scale, consistent direction across all four
   sub-dimensions), but we want to verify the bar before treating substring
   grading as "validated for this corpus."

3. **Is the Mistral v4 < Mistral stock judge signal (Finding 2) too weak to
   register at all,** or weak corroboration that supports continuing with
   Task 4 (Llama v4) as a more cleanly independent test? Asking because the
   reviewer's framing rules are quite clear about variance bounds being the
   gate; want to confirm we're not over-weighting the descriptive judge data.

---

## Caveats

- **Partial sample counts** (n=18-42 across models) due to early-run rate
  limiting. The 18-sample v3 result is the worst case; v1 had 34, others 39-42.
  Same deterministic prompt selection across runs, so partial overlaps are
  comparable on the prompts that succeeded for both.
- **0% cache hit rate** — system prompt was below Sonnet 4.6's 2048-token
  cache minimum. Easy fix (expand rubric examples) for any future runs but
  doesn't affect data quality.
- **Two judge crashes** (Pydantic ValidationError on Qwen 32B first run, also
  some non-fatal validation failures absorbed by the patched try/except).
  Sonnet 4.6 occasionally returns scores outside 0-2 despite the schema
  spec — schema-strict mode would catch these but doesn't recover from them.

---

*Generated 2026-05-08 by [docs] session as evidence-dump for review. No
gotcha #7 doc commits proposed. Tasks 3 + 4 (variance bounds + Llama v4)
remain the gating items per the reviewer's halt directive.*
