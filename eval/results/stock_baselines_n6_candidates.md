# N=6 candidate stock baselines — Phi-3-mini, Yi-1.5-9B-Chat, Gemma 2 2B

**Date:** 2026-05-09
**Trigger:** reviewer Q3 hybrid recommendation — measure stock reasoning floors on candidate bases before fine-tuning, so the N=6 candidate is selected by data not by guess.
**Apples-to-apples:** all three runs at temp=0, RAG=on, `eval/prompts_v2.json`, 132-sample basis (post-regrade 126 with persona quarantined). Same eval pipeline as the existing N=5 baselines.

## Summary

| Base | Stock total (post-regrade /126) | Stock reasoning /6 | Stock refusal /9 | Verdict for N=6 selection |
|---|---:|---:|---:|---|
| Phi-3-mini-4k-instruct | 12/126 = 9.5% | 3/6 | 0/9 | **Disqualified — 4K context window saturated by RAG (rag_datasheet 0/78). Fine-tune comparison would be context-broken vs context-broken, not stock-vs-FT.** |
| Yi-1.5-9B-Chat | 86/126 = 68.3% | 3/6 | 6/9 | **Best candidate.** Solid stock baseline, intermediate reasoning band, no context handicap. Different family (01.AI). |
| Gemma 2 2B-it | 72/126 = 57.1% | 3/6 | 6/9 | Viable as a size-confound point. Same family as our existing Gemma 2 9B v4 result, smaller param regime. |

**All three candidates landed at 3/6 reasoning** — the same intermediate band as Qwen 14B (the existing N=5 intermediate-band data point). No 4/6 or 5/6 candidate emerged from this trio. The 4/6 and 5/6 reasoning bands remain uncharacterized after this measurement round.

## Per-category breakdowns

### Phi-3-mini-4k-instruct stock

| Category | Pass /n |
|---|---:|
| coding | 6/6 |
| general | 0/6 |
| multihop | 0/9 |
| numerical_precision | 0/6 |
| rag_blog | 3/3 |
| rag_datasheet | **0/78** ← context-window failure |
| rag_email | 0/3 |
| reasoning | 3/6 |
| refusal | 0/9 |

### Yi-1.5-9B-Chat stock

| Category | Pass /n |
|---|---:|
| coding | 4/6 |
| general | 3/6 |
| multihop | 6/9 |
| numerical_precision | 6/6 |
| rag_blog | 3/3 |
| rag_datasheet | 55/78 |
| rag_email | 0/3 |
| reasoning | 3/6 |
| refusal | 6/9 |

### Gemma 2 2B-it stock

| Category | Pass /n |
|---|---:|
| coding | 6/6 |
| general | 3/6 |
| multihop | 5/9 |
| numerical_precision | 6/6 |
| rag_blog | 3/3 |
| rag_datasheet | 37/78 |
| rag_email | 3/3 |
| reasoning | 3/6 |
| refusal | 6/9 |

## Updated N-per-band picture

With the three candidates measured, the N-per-band visibility on stock reasoning is now:

| Reasoning band | N | Bases | Action |
|---|---:|---|---|
| 0/6 | 1 | Mistral 7B v0.3 | regressed on substring + judge (existing) |
| 1/6 | 1 | Llama 3.1 8B | regressed on substring + judge (existing) |
| 2/6 | 0 | — | uncharacterized |
| 3/6 | **4** | Qwen 14B, Phi-3-mini-4k, Yi-1.5-9B-Chat, Gemma 2 2B-it | Qwen 14B fine-tune lifts on substring (judge erases); the other three are candidates for N=6 if we want a different-family confirmation |
| 4/6 | 0 | — | uncharacterized |
| 5/6 | 0 | — | uncharacterized |
| 6/6 | 2 | Qwen 7B, Gemma 2 9B | both lifted on substring (judge erases) |

**For N=6 fine-tune (when we actually run it):**

- **Best candidate: Yi-1.5-9B-Chat.** Solid stock baseline (68.3%), 3/6 reasoning, different family (01.AI), no context-window handicap. A fine-tune would test "does the v4 recipe lift on a *different* family at 3/6 reasoning, the way Qwen 14B did?"
- **Disqualify: Phi-3-mini-4k-instruct.** 4K context window is too small for the v2-rag eval; the FT vs base comparison would be confounded by context-window saturation, not capability transfer.
- **Optional secondary: Gemma 2 2B-it.** Same family as the existing Gemma 2 9B lifter; would test size-confound (does the v4 recipe still lift at smaller param counts within a family that lifted at 9B?). Useful but not the highest-information data point.

**Bands still uncharacterized:** 2/6, 4/6, 5/6. To fully characterize the predictor, we'd need at least one base in each of those bands. That's downstream work, not blocking customer-template publication.

## Files / artefacts

- `eval/results/acc_baseline-phi-3-mini-4k-instruct-v2-rag_20260509-183400.json`
- `eval/results/acc_baseline-yi-1.5-9b-chat-v2-rag_20260509-183837.json`
- `eval/results/acc_baseline-gemma-2-2b-it-v2-rag_20260509-184055.json`
- `models/phi-3-mini-4k-hf/phi-3-mini-4k-instruct-stock-q4_k_m.gguf` (2.3 GB)
- `models/yi-1.5-9b-chat-hf/yi-1.5-9b-chat-stock-q4_k_m.gguf` (5.0 GB)
- `models/gemma-2-2b-it-hf/gemma-2-2b-it-stock-q4_k_m.gguf` (~1.6 GB)

All pushed (or pushing) to `gdrive:skippy_files/personal-ai-assistant/`.

Production llm-server temporarily swapped through Phi → Yi → Gemma 2 2B for the eval cycle, then restored to 7B v4 (verified healthy).
