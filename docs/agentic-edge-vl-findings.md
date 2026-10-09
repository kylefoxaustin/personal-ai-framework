# Agentic-Edge VL Capstone — the perception workload (skunk-detector benchmark)

Post-Friday companion to the LLM deck. The 10 LLM tasks are decode/bandwidth-bound with the NPU
idle for the loop; this is the **one genuinely NPU/accelerator-bound agentic task** — perception —
so it closes the resource story. A Qwen2.5-VL-7B "find the animal" benchmark on **splat-vla's real
24/7 backyard-camera corpus** (377 clips, 8 nights, a free classical-CV baseline on identical
inputs, human-confirmed labels). splat-vla ran the measurements; this doc synthesizes them and owns
the resource analysis.

## HEADLINE (resource) — resolution turns an LLM-shaped job into an NPU-shaped one
Measured on the 5090 (splat-vla's run); aggregation here. As input resolution climbs, image tokens
grow, **prefill (the vision encode) comes to dominate**, while decode stays resolution-flat:

| rung | img+prompt tok | prefill t/s | decode t/s | prefill ms | decode ms | **prefill share** |
|---|--:|--:|--:|--:|--:|--:|
| 0.25 MP | 673 | 5633 | 175.6 | 55 | 202 | **21.5%** |
| 0.50 MP | 1007 | 6080 | 173.8 | 106 | 212 | **33.3%** |
| 1.00 MP | 1657 | 5404 | 173.5 | 239 | 232 | **50.8%** |
| native 4.09 MP | 4441 | 3123 | 167.8 | 1409 | 226 | **86.2%** |

**This is the thesis, completed.** The LLM agent loop is decode-bound (bandwidth, the NPU idle);
a VL perception front-end at real resolution is **86% prefill** — compute/TOPS-bound, *exactly where
the NPU earns its area*. Resolution is the knob that moves work between the two budgets. Decode is
flat (~170 t/s) because it's the same short text generation regardless of image size. n=204/rung.
Provenance: per-frame rates MEASURED (splat-vla, 5090, Q8_0, PTX-JIT sm_90→sm_120 so a lower bound);
the table aggregation DERIVED here. Confirms splat-vla's reported 23%→88%.

## ACCURACY (splat-vla's measurements + methodology — cite, don't re-derive)
splat-vla designed the labeling protocol + its controls; these are their numbers, with their
caveats. Cross-checked here for shape against the raw data (`human_confirmed.csv`, `set_a_*.jsonl`).
> ⏳ Exact accuracy figures pending splat-vla's full-set restatement (denominator: species-only vs
> all-verdict; `unsure` handling). Shape below is confirmed; precise % to be locked with them.

- **Detection — the VLA structurally beats the CV baseline.** 127 VLA-only detections the motion
  baseline missed, mostly **motionless perched birds a motion detector physically cannot see**. This
  is a mechanism win, not tuning. Licensed as a **precision** figure on the VLA-flagged subset
  (verification-with-prior labeling, anchoring-controlled: 18/20 with box = 18/20 without) — **not a
  recall claim**, not blind ground truth.
- **Species — report ceiling-relative.** Human intra-rater self-agreement is the ceiling (~69–75%
  depending on subset; the human changed 30.7% of verdicts on blind re-review). Model species ≈
  61–66% raw → ~88% *of the human ceiling*, not "33 points off perfect." Dominant error: confusing
  small birds with raccoons/squirrels (seen directly in `human_confirmed.csv`).
- **Direction-of-travel — a clean VLA win** (the CV baseline can't do it at all); scored only on
  pairs passing a declared correspondence gate, `INDETERMINATE` otherwise with the fraction published
  — no fabricated threshold.
- **Resolution recovers 1/3 of the recall gap.** Native vs 0.25 MP (same frames/prompt) recovers 35%
  of the missed animals at 3% new false-fires — i.e. a *third* of the failure is resolution, not
  reasoning, and the fix costs 7.3× wall + the prefill-share jump above. On a fixed accelerator
  budget, **buy pixels before a bigger model** is the cheaper lever.

## The conclusion (splat-vla's, and it's the right shape)
**Do not put a 7B VL model alone behind the camera.** Cheap classical CV as the recall gate; the VLA
for what motion structurally cannot do — **species, direction, and the still-animal case**. A hybrid,
and more lopsided in the hybrid's favor than predicted.

## Caveats that travel with every number
Q8_0 (not Q4); PTX-JIT sm_90→sm_120 so throughput is a LOWER BOUND flattering the 5090; one camera /
one yard / 8 nights; detection from the 0.25 MP rung; one human labeller with stated reliability
bounds; verification-with-prior (precision, not recall); one `nothing`-labelled subset is
hand-contaminated and must not be reused as species ground truth.

## Next — the embodied-VLA extensions (to scope with qualcomm)
The skunk benchmark is *perception* (classify/locate). The VP's "a cat's in the yard — find it, show
me a picture" generalizes to **embodied** VLA — prompt → perceive → **act**:
- **GR00T virtual robot** ("pick up that object") and **drone nav** ("fly to the red post") — qualcomm
  has GR00T + smolVLA running on the iq9 (anchor data on the edge NPU already). Same resource shape
  expected: perception/prefill compute-bound (NPU), the policy/decode loop light.
- To scope with qualcomm once their iq9 cDSP incident clears: a standardized prompt→action task +
  the per-phase (perception vs policy) resource split, so it joins this ladder.
