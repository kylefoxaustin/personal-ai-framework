# Agentic-Edge VL Benchmark — Prompt + Output Schema (v1, 2026-10-07)

Protocol for the skunk-detector VL benchmark (Skippy authors, splat-vla runs the corpus).
Qwen2.5-VL-7B vs the classical-CV baseline on splat-vla's backyard-camera corpus. Drafted
against real frames (045852 night-IR skunk-under-bench + human-hand distractor; 161651 daylight
small-bird not-a-skunk case), so the prompt's IR / clutter / human-exclusion clauses are grounded.

## Inference unit
**One frame per inference** (single image). This keeps each call a clean single-image prefill so
the per-resolution prefill/decode cost is measured without multi-image confound. Direction-of-travel
is DERIVED post-hoc from the per-frame bbox centroids across consecutive frames (SET B, 2 fps gives
the consecutive pairs) — not asked of the model per frame, so no extra image prefill.

## Prompt (fixed; identical across every board and resolution rung)
```
You are an animal-detection vision system for a backyard security camera labeled "LEFT SIDE YARD".
Images may be daytime color OR nighttime monochrome infrared, wide-angle, and cluttered with
furniture, gravel, plants, toys, and sometimes a human hand or body.

Report ONLY non-human animals (for example: skunk, cat, dog, raccoon, opossum, bird, deer, rodent).
Do NOT report humans, human hands/limbs, or inanimate objects as animals.

Look carefully. Animals may be partially hidden under or behind objects, and may be small or
low-contrast in infrared.

Respond with STRICT JSON ONLY — no prose, no code fence — matching exactly:
{"animal_present": <true|false>, "species": "<skunk|cat|dog|raccoon|opossum|bird|deer|rodent|other_animal|none>", "bbox": <[x0,y0,x1,y1] normalized 0-1 of the single most prominent animal, or null>, "facing": "<left|right|toward_camera|away|unclear|none>", "confidence": <number 0.0-1.0>}

If no animal is present: animal_present=false, species="none", bbox=null, facing="none".
```

## Output schema (what the grader parses)
| field | type | scored on | notes |
|---|---|---|---|
| `animal_present` | bool | **SET A** (detection, vs CV baseline) | the head-to-head |
| `species` | enum | **SET B** (classification) | CV can't do this at all |
| `bbox` | [x0,y0,x1,y1] 0-1 \| null | localization + feeds direction | normalized so resolution-invariant |
| `facing` | enum | — (diagnostic) | pose cue, not the travel direction |
| `confidence` | 0-1 | thresholding / PR on SET A | |
| *direction-of-travel* | derived | **SET B** | from bbox-centroid delta across consecutive frames |

## Grading provenance (LAW 1, per splat-vla's split)
- SET A = human-confirmed labels → **MEASURED** scored set; scores `animal_present` detection vs CV.
- SET B = Opus-read labels → **DERIVED** agreement set; scores `species` + direction ONLY;
  `never_score=[detection, precision]` (no detection/precision on a fabricated prior). Reported
  separately, never mixed into one number.

## Open items for splat-vla to confirm
1. Species enum coverage — does the corpus have classes beyond this list (opossum/armadillo/fox)?
   Add any real ones; keep `other_animal` for the long tail.
2. bbox origin convention: top-left (0,0) → bottom-right (1,1), x then y. Confirm it matches the
   CV baseline's box convention so SET A localization is comparable.
3. Direction derivation: centroid-delta sign over the consecutive pair, threshold below which it's
   "stationary". Propose the threshold by rule before running.
