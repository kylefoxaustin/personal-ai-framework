# Runbook: stock-baseline measurement for N=6 candidate selection

**Status:** staged, not executed. Reviewer-blessed (Q3) and gated on Kyle's go to download base models.

## Purpose

Identify a base whose stock reasoning floor lands at 3–4/6 (intermediate band) on the v2 eval set. This is the highest-information next data point to falsify or confirm the reasoning-floor predictor at N=6.

Per the reviewer's Q3 verdict, we **must measure** before fine-tuning. Picking blind risks burning a fine-tune cycle on a base that ends up at 6/6 (predicted-lifter band; already characterized by Qwen 2.5 7B/14B and Gemma) or 0–1/6 (predicted-regress band; already characterized by Mistral and Llama).

## Candidate bases

1. **Phi-3-mini-4k-instruct** (Microsoft, 3.8B). Different family from anything in N=5 (Microsoft, distinct from Qwen / Mistral / Llama / Gemma). Smaller param regime — tests size as a confound. **Reviewer's blind preference** if we couldn't measure first.
2. **Yi-1.5-9B-Chat** (01.AI, 9B). Different family; size in the same band as Gemma 9B for clean comparison.

## Estimated cost

Local 5090 only. ~1–2h per model end-to-end (download + GGUF conversion + eval). No external spend.

## Procedure per candidate

### 1. Download HF base

```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
  --local-dir models/phi-3-mini-4k-hf
huggingface-cli download 01-ai/Yi-1.5-9B-Chat \
  --local-dir models/yi-1.5-9b-chat-hf
```

Yi may require accepting license on the HF web UI first (similar to Llama / Gemma access flow).

### 2. Convert to GGUF Q4_K_M

(Matches the rest of the N=5 set; apples-to-apples eval.)

```bash
# Phi-3-mini
python <llama.cpp>/convert_hf_to_gguf.py models/phi-3-mini-4k-hf \
  --outfile models/phi-3-mini-4k/phi-3-mini-4k-stock-f16.gguf --outtype f16
<llama.cpp>/llama-quantize \
  models/phi-3-mini-4k/phi-3-mini-4k-stock-f16.gguf \
  models/phi-3-mini-4k/phi-3-mini-4k-stock-q4_k_m.gguf q4_k_m

# Yi-1.5-9B
python <llama.cpp>/convert_hf_to_gguf.py models/yi-1.5-9b-chat-hf \
  --outfile models/yi-1.5-9b/yi-1.5-9b-chat-stock-f16.gguf --outtype f16
<llama.cpp>/llama-quantize \
  models/yi-1.5-9b/yi-1.5-9b-chat-stock-f16.gguf \
  models/yi-1.5-9b/yi-1.5-9b-chat-stock-q4_k_m.gguf q4_k_m
```

(Adjust `<llama.cpp>` to the actual checkout path on this machine.)

### 3. Swap into `pipeline/config.yaml`

`pipeline/config.yaml` is gitignored — local change only. Restart `llm-server` after each swap.

### 4. Run accuracy eval

Apples-to-apples — same v2 prompts at temp=0, RAG-on, 132-sample basis.

```bash
python eval/run_accuracy_eval.py \
  --base-url http://localhost:8080 \
  --auth kyle:123456 \
  --output eval/results/acc_baseline-phi-3-mini-4k_$(date +%Y%m%d-%H%M%S).json
```

(Repeat for Yi.)

### 5. Inspect the reasoning category specifically

```bash
python eval/compare_accuracy_runs.py eval/results/acc_baseline-phi-3-mini-4k_*.json --per-category
```

Note: `reasoning` category is 2 prompts × 3 samples = 6/6 max.

### 6. Restore production 7B v4

Set `pipeline/config.yaml` back to `kyle-7b-v4-q4_k_m.gguf` and restart `llm-server`. Verify `/health` returns 7B v4 before declaring done.

### 7. Push baseline JSONs to Drive

Per artifact-auto-push rule:

```bash
rclone copy eval/results/acc_baseline-phi-3-mini-4k_*.json \
  gdrive:skippy_files/personal-ai-assistant/eval-results/
rclone copy eval/results/acc_baseline-yi-1.5-9b-chat_*.json \
  gdrive:skippy_files/personal-ai-assistant/eval-results/
```

(Stock GGUFs can also be pushed if disk pressure becomes an issue, but per `feedback_preserve_failure_data.md`, do NOT delete them locally.)

## Decision gate (after both measurements)

| Outcome | Action |
|---|---|
| **One lands at 3–4/6 reasoning** | That base is the N=6 candidate. Proceed to fine-tune (separate runbook; pattern matches `docs/runpod-tier2-handoff.md` style for cross-family runs). |
| **Both land at 0–1/6 or 5–6/6** | Neither tests the intermediate band. Identify a different base (candidates: Phi-3.5-MoE, Yi-1.5-34B-Chat, Mistral-Nemo-12B). Re-run this procedure on the new candidate. |
| **Mixed (one in band, one out)** | Use the in-band one for N=6. Optionally fine-tune the out-of-band one as a confirming data point (low priority). |

## What NOT to do

- Do NOT skip the stock-baseline step and start fine-tuning on either candidate. The whole point of N=6 is to test the intermediate band — picking blind risks burning a fine-tune on a base that lands at ceiling or floor (already characterized).
- Do NOT delete the stock GGUFs after the measurement (per `feedback_preserve_failure_data.md`).
- Do NOT skip restoring production 7B v4 to `pipeline/config.yaml` after the measurements — the llm-server must be back on production before any other work resumes.

## Post-actions

- Update `docs/recipe-taxonomy.md` Tier 3 dispatch table with the chosen N=6 cell.
- Bus message to [backend] with the band-membership decision.
- If the chosen base is intermediate, link this runbook from `docs/REVIEWER_UPDATE_N5.md` Holds Released so the reviewer can track the N=6 sequence.
