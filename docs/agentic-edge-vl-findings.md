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

## ACCURACY — final figures (MEASURED by splat-vla; cited verbatim with their denominators/caveats)
splat-vla designed the labeling protocol + its controls; these are their locked numbers.

- **Detection is a PRECISION story, not a recall one.** On the frames the VLA flagged, precision is
  **96.7% (117/121, excluding 6 `unsure`; 92.1% if `unsure` counts as failures)**, anchoring-controlled
  (18/20 with the model's box = 18/20 without). It is NOT a detection rate. The VLA's *recall* is the
  weak leg — a **4.1× loss** vs the task. What it uniquely catches is the **still-animal case**: 127
  motionless perched birds a motion baseline physically cannot see. The value is **complementarity**
  (precise + sees stillness), not "beats CV on recall."
- **Species: 66.7% (78/117) against a 74.8% intra-rater ceiling (83/111)** — denominator = frames the
  human labelled an actual animal (excludes `nothing` and `unsure`). That's ≈89% of the human ceiling,
  not "33 points off perfect" (the human changed 30.7% of verdicts on blind re-review).
- **Species errors — the asymmetry is in the CONSEQUENCE, not the frequency.** Of 39 errors,
  bird→mammal 19 (49%) and bird→another-bird 18 (46%) are **tied**. But the mammal confusions are the
  deployment-relevant ones: **6 raccoons and 1 skunk called "dove," at night in IR, on a camera whose
  job is finding skunks.** A dove/cardinal mixup costs nothing; a skunk-called-dove is the failure.
- **Direction-of-travel — NO accuracy figure exists.** No human direction labels were collected, so
  any direction-*accuracy* number in a writeup is **fabricated**. Publish only the **indeterminate
  fraction: 69.7% @0.25MP, 77.7% @native (n=3340 consecutive pairs)**, of which ~2/3 is upstream
  detection failure, not the correspondence gate (2.9%).
- **Resolution recovers 1/3 of the recall gap — but 2/3 survives.** Native vs 0.25MP (same frames)
  recovers **13/37 = 35% (95% CI 20–51%)** of the missed animals at **1/43 = 2%** new false-positives
  (n=81, all native verdicts). **Residual: 65% of the misses survive native resolution** — resolution
  is a cheap *partial* lever (and it costs the 7.3× wall + prefill-share jump above), not a fix. Buy
  pixels before a bigger model, but know the ceiling.

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
