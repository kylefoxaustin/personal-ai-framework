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

## Prompt (v1.1 — fixed; identical across every board and resolution rung)
Species enum is CORPUS-DRIVEN (splat-vla surveyed all 17 positive clips): classes actually present
are skunk / dog / deer / bird (3 visually-distinct: male cardinal, female cardinal, dove) / human /
nothing. `human` is a SCORED NEGATIVE CONTROL, not an exclusion — it's a large moving non-animal (the
hard negative the CV motion-baseline falsely fires on). raccoon/opossum/fox are NOT in the corpus;
kept only as fallback values, never reported as "supported classes."
```
You are an animal-detection vision system for a backyard security camera labeled "LEFT SIDE YARD".
Images may be daytime color OR nighttime monochrome infrared, wide-angle, and cluttered with
furniture, gravel, plants, toys, and sometimes a person or a hand/arm reaching into frame.

Identify the single most prominent subject. A HUMAN or human body part is NOT an animal: if the only
subject is a person, set animal_present=false and species="human" — do not call it an animal.

Look carefully. Animals may be partially hidden under or behind objects, and may be small or
low-contrast in infrared. Distinguish bird types when you can (a brilliant red bird is a male
cardinal; an olive-brown crested bird is a female cardinal; a plain gray-brown bird is a dove).

Respond with STRICT JSON ONLY — no prose, no code fence — matching exactly:
{"animal_present": <true|false>, "species": "<skunk|dog|deer|bird_cardinal_male|bird_cardinal_female|bird_dove|bird_other|human|other_animal|none>", "bbox": <[x0,y0,x1,y1] normalized 0-1 of the single most prominent subject, or null>, "facing": "<left|right|toward_camera|away|unclear|none>", "confidence": <number 0.0-1.0>}

Rules: animal_present=true ONLY for a non-human animal. For a human-only frame: animal_present=false,
species="human". For neither animal nor human: animal_present=false, species="none", bbox=null.
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

## Resolved (v1.1, 2026-10-07 — after splat-vla's corpus survey)
1. **Species enum:** corpus-driven (above). Fine-grained bird classes kept (the corpus's only
   fine discrimination — male vs female cardinal is one species, two appearances). `human` is a
   scored negative control (animal_present MUST be false on it). raccoon/opossum/fox absent → not
   reported as supported.
2. **bbox / localization:** bbox stays a required model output (also feeds direction). splat-vla
   adds the CV largest-connected-component (background-subtraction) box, and we report IoU as a
   clearly-labeled SECONDARY "agreement with the motion baseline" metric on clean single-component
   frames — NOT an object-localization-accuracy claim (a motion box ≠ an object box).
3. **Direction-of-travel:** NO fabricated threshold — the 2 fps displacement distribution is
   continuous over 3 orders of magnitude with no stationary/moving gap, so any constant would
   invent the scores (LAW 1). Instead: score direction only on consecutive pairs passing a declared
   CORRESPONDENCE gate (both frames have a >1500 px component, area ratio within 2×); emit
   INDETERMINATE otherwise and PUBLISH the indeterminate fraction. Positive clips re-cut at 8 fps
   (17 clips, cheap) so displacement/step drops ~4× and correspondence is safe; direction scored
   primarily on the 5090. A model that is honestly uncertain is not punished by our gate.
4. **SET B `none` labels:** SET B membership is a MOTION criterion (gap≥12%), not an animal one, so
   it contains non-animal motion events (e.g. 070823 has no visible animal) + the 3 human clips.
   These get real `none`/`human` labels and become the FALSE-POSITIVE test: did the VLA correctly
   withhold animal_present on motion that isn't an animal — exactly where the CV baseline fails.

Status: **v1.1 FROZEN** pending splat-vla's go on the 8 fps re-cut + CV-bbox add.
