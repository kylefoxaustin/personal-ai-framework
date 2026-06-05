# NPU Precision-Set Selector — cross-surface design spec

**Status:** ratchet ✅ v0.2.7 · PAI-sizer ✅ built+validated · keyhole-sizer mirror pending (2026-06-05)
**Author:** Skippy / [docs] (cross-surface spec authority, per CLAUDE.md)
**Consumers:** ratchet (engine, done), PAI sizer (UI, done), keyhole-sizer (mirror, building)
**Routed to:** [ratchet], [pai-sizer] via Claude Bus 2026-06-05

## 1. Goal

Let a PAI-sizer user pick an NPU tier (Mid / High) **and independently
select its precision capability** as an escalating ladder:

```
  INT-only   →   INT + FP8   →   INT + FP8 + FP4
```

The purpose is to let Kyle (and customers) **see the benefit** of FP-capable
edge silicon by A/B/C-ing the same model on the same memory class across the
three precision states and reading off the TTFT / TPS / RAM deltas.

## 2. Core principle — precision is orthogonal to tier

Today ratchet conflates *memory/compute class* with *precision capability* into
fixed silicon classes (`neutron`, `lp5x_128_int8`, `lp5x_128`, …). This spec
**separates the two axes**:

- **Tier axis** (`silicon_class`): memory class + base TOPS + calibration.
  Stays as-is — Low / Mid / High map to memory classes.
- **Precision axis** (`npu_precision_set`, NEW): which dtypes the tensor
  engine executes natively. This is the user's checkbox/radio.

The two compose. "Mid · INT+FP8" = Mid memory class × FP8-capable engine.

## 3. Why this is an escalating ladder, not free checkboxes

No real silicon ships FP4 datapaths without FP8 — FP4 MACs are built on top of
the FP8 ones. So the selector is a **3-step radio**, not 2³ independent
checkboxes. Valid states only:

| State | int8 | fp8 | fp4 |
|---|:---:|:---:|:---:|
| `int8` | ✅ | — | — |
| `int8_fp8` | ✅ | ✅ | — |
| `int8_fp8_fp4` | ✅ | ✅ | ✅ |

### 3.1 The rung SETS the model's compute_dtype (load-bearing semantic)

**Verified against ratchet `dtype_map.py` (2026-06-05):** a Q4_K_M model carries
`compute_dtype='fp16'`, which routes to `peak_tops_bf16`. So a *naive* Q4 run
reads the bf16 field — NOT int8/fp8. If the precision-set rung did not change
this, the FP8 rung would read bf16 (200 on High) and show **zero prefill
benefit** — the selector would be inert.

Therefore the rung **sets the effective compute_dtype** for the projection:
`int8 → 'int8'`, `int8_fp8 → 'fp8'`, `int8_fp8_fp4 → 'nvfp4'`. This is a
*runtime-capability assumption*: "the vendor runtime executes the matmul at the
selected precision." For int8 (W8A8) and fp8 that assumption is mature/standard;
for fp4 it is shaky → handled by the maturity sub-axis (§6). The honest
**comparison baseline** is the naive Q4/fp16 run (reads bf16 TOPS).

## 4. TOPS ladder (datapath-doubling physics)

Per the 2026-04-29 silicon-physics lock: each precision halving doubles MAC
density on the same datapath. 16-bit = X, 8-bit = 2X, 4-bit = 4X. FP8 and INT8
share the 8-bit datapath → **same TOPS** (FP8 is not faster than INT8 — but both
are 2× the bf16 field).

| Tier | bf16 | int8 | fp8 | fp4 |
|---|---:|---:|---:|---:|
| **Mid** (FP-capable) | 100 | 200 | 200 | **400** |
| **High** (FP-capable) | 200 | 400 | 400 | **800** |

Note the two ratios that earlier drafts conflated: High fp4 = **4× the bf16
field** (800 vs 200) = **2× the fp8/int8 path** (800 vs 400). Both true — §5
states speedups against the bf16 baseline (the naive-Q4 reference), §9 anchors
the FP4 rung against the fp8 path.

Mid stock today is INT8-only (no FP hardware at all); "Mid · INT+FP8/FP4" posits
an FP-capable chip at the Mid memory class. FP4 TOPS (400 Mid / 800 High) is a
🟠 **modeled projection** — zero FP4 silicon anchors exist on any NPU. Flag as
`confidence='low'` / cross-class until a vendor FP4 measurement lands.

## 5. The benefit story each rung buys (the whole point of the tool)

Measured against the **naive-Q4/fp16 baseline** (§3.1) — i.e. what a stock
llama.cpp-style dequant-to-fp16 run gives, reading the bf16 field:

- **naive Q4 (fp16 dequant) → INT-only (W8A8): buys 2× PREFILL, costs accuracy.**
  int8 is 8-bit = 2× the bf16 datapath, so prefill halves. But we *measured*
  INT8 W8A8 at **−3.8pp** (reproducible, refusal-specificity) — this is the
  accuracy cliff edge INT8-only silicon forces.
- **INT-only → +FP8: buys ACCURACY, holds the speed.** FP8 == INT8 in TOPS
  (same 8-bit datapath), so prefill is **unchanged vs INT8** — still 2× the
  fp16 baseline. What changes is fidelity: FP8 is near-lossless, so this rung
  **recovers the −3.8pp INT8 cliff at zero speed cost.** FP8's advantage is
  *purely* the accuracy recovery over INT8 — the rung most likely to be
  undersold.
- **+FP8 → +FP4: buys MORE PREFILL SPEED (and accuracy between).**
  - *Prefill / TTFT (compute-bound):* 2× the 8-bit TOPS → **4× the fp16
    baseline / 2× the fp8 path**. Matters for RAG (long stuffed prompts) —
    Skippy's exact shape. Subject to the maturity caveat (§6).

### 5.1 RAM is an ORTHOGONAL axis — NOT a compute-rung benefit

⚠️ **Corrected 2026-06-05** (pai-sizer + backend, ratified by [docs]). Earlier
drafts wrongly bundled "half RAM" into the FP4 *compute* rung. They are two
independent axes:

| Axis | What it is | What it changes |
|---|---|---|
| **Compute rung** (this selector) | int8 / fp8 / nvfp4 MATH path | prefill speed + accuracy |
| **Weight format** (model choice) | Q4 / Q8 / FP4-weight bytes stored | RAM + decode BW |

For an **already-Q4 model** the weights are 4-bit at *every* compute rung, so the
selector changes prefill + accuracy but leaves **weight RAM fixed**. The half-RAM
unlock (14B fitting ~7 GB) only appears if you change the *weight format*
(deploy FP4-weight vs INT8-weight) — a separate deployment choice, not the
compute rung. Conflating them re-introduces the exact memory-format-vs-compute-
format confound the whole FP4 story exists to separate (ADR-016 / the
INT4-as-bf16-floor asymmetry). The compare panel renders RAM as fixed-by-model
with a caption pointing to the weight-format axis.

*Decode / TPS (bandwidth-bound):* unchanged by the compute rung for a Q4 model
(BW-bound on the same 4-bit bytes). Decode only moves if the *weight format*
changes — and on an immature FP4 runtime, ADR-016 says FP4-weight can be
**15–19% slower** than Q4_K_M. See §6.

**One-line headline:** INT8 and FP8 *both* buy 2× prefill over a naive fp16 run;
FP8's edge over INT8 is the accuracy recovery; FP4 buys 4× prefill. RAM is a
separate weight-format axis, not a compute-rung effect.

## 6. Runtime-maturity sub-axis (ADR-016)

FP4's *prefill compute win is runtime-conditional*. Same NVFP4 weights, same
5090: vLLM gave 2.24× decode / 3.59× prefill; llama.cpp was 15–19% slower than
Q4_K_M. Edge NPU vendor runtimes are custom (not vLLM), so maturity is
**unknown per-vendor**.

Therefore:
- When `int8_fp8_fp4` is selected, the UI MUST reveal a **mature / immature**
  runtime toggle.
- **Edge default = `immature`** (honest floor: FP4 modeled as INT4 weight-only,
  prefill falls to the bf16 floor, decode stays BW-bound on 4-bit bytes). Only
  flip to `mature` if the target vendor runtime is proven vLLM/TensorRT-class.

## 7. ratchet changes — ✅ SHIPPED in v0.2.7 (ADR-017)

Implemented and tagged `@v0.2.7` (origin/main `69a17bf`, tag `2c99f22`, 245
tests). As built:

1. **`NPU_FP4_CAPABILITY`** = `NPU_FULL` + `nvfp4 = TENSOR_NATIVE`, forward-looking
   / zero-anchors / confidence-low. Kept `TENSOR_NATIVE` (not `tensor_compat`):
   capability = silicon per ADR-008/016; the over-promise is fenced by the
   maturity axis, not by downgrading the level.
2. **New `Hardware.npu_precision_set` field** + `make_custom_tier` param:
   `'int8' | 'int8_fp8' | 'int8_fp8_fp4' | None`. The rung is **atomic** — set on
   the tier, the projection reads it; building the tier makes prefill correct,
   no second arg to forget. It overrides the model's `compute_dtype` for the
   floor (int8/fp8/nvfp4), zeros `peak_tops_*` above the rung, and selects the
   capability dict (`PRECISION_SET_CAPABILITY`). Chosen as a *new* field (not
   derived from `capability_levels`) because deriving would silently flip
   existing NPU_FULL/High projections fp16→fp8 = breaking; `None` on canonical
   tiers = non-breaking.
3. **`fp4_runtime_maturity`** stays a `project_llm` param (runtime ≠ silicon),
   engine default `'mature'`. The edge-default-`immature` is pai-sizer POLICY
   (§6), so pai-sizer MUST pass it explicitly.
4. New exports: `NPU_FP4_CAPABILITY`, `PRECISION_SET_CAPABILITY`,
   `NpuPrecisionSet`, `resolve_floor_dtype`.

## 8. pai-sizer changes (UI — ✅ UNBLOCKED, build now)

Shipped API:
```
make_custom_tier(name, silicon_class='lp5x_128_int8'(Mid)|'lp5x_128'(High),
    peak_tops_int8=, peak_tops_fp8=, peak_tops_fp4=, peak_tops_bf16=, mem_*=...,
    npu_precision_set='int8'|'int8_fp8'|'int8_fp8_fp4')
project_llm(model, tier, workload, prompt_tokens=,
    fp4_runtime_maturity='mature'|'immature')
```
- Bump ratchet pin to `>=0.2.7`.
- Escalating radio under the Mid/High selector: INT-only / +FP8 / +FP8+FP4 →
  maps 1:1 to `npu_precision_set`.
- Reveal mature/immature toggle when +FP8+FP4 is chosen; **default `immature`,
  passed explicitly** (engine default is `mature`).
- Render TTFT / TPS / RAM across the selected state; ideally a small 3-column
  compare so the benefit deltas are legible at a glance.

## 9. Validation anchors (must reproduce)

All prefill anchors assume the rung sets compute_dtype (§3.1). ✅ **Reproduced
exactly on PAI-sizer (ratchet v0.2.7), Qwen3-30B-A3B MoE @1K, npu_share=1.0.**

**Baseline is MEASURED-anchored, not computed.** The bf16 floor (naive Q4/fp16
and the immature-FP4 collapse) anchors to the **real measured Mid cell = 351 ms**,
NOT a first-principles compute floor (333 ms). Ratified 2026-06-05 by [docs] on
pai-sizer's + backend's rec: "measured, not modeled" is the discipline, 351 is on
silicon, and it makes the relationships *exact* (naive→INT8 = precisely 2×,
immature-FP4 == naive exactly). The ~5% gap is the modeled-vs-measured delta,
resolved toward measured.

| Anchor | dtype read | Expected | Note |
|---|---|---|---|
| **High · naive Q4 (fp16)** | bf16 | **351 ms** | baseline (bf16=200 TOPS == Mid int8) |
| High · INT-only (int8) | int8 400 | **175.5 ms** | exactly 2× the fp16 baseline; −3.8pp acc |
| High · +FP8 (fp8) | fp8 400 | **175.5 ms** | == INT8 speed, near-lossless acc |
| High · +FP4 mature (nvfp4) | fp4 800 | **87.8 ms** | 4× baseline / 2× the fp8 path |
| High · +FP4 immature | bf16 floor | **351 ms** | ADR-016 collapse → == naive (honest no-win) |
| Mid · INT-only (int8) | int8 200 | **351 ms** | measured Mid cell (2026-04-29) |
| Mid · +FP8 (fp8) | fp8 200 | **351 ms** | == INT8 speed on Mid |
| Any tier · decode @100% share | — | **37.9 t/s** | BW-bound, model already Q4 — unchanged |

(The 351 / 175.5 pair only reproduces if the rung sets compute_dtype to
int8/fp8 — confirming §3.1 is load-bearing, not cosmetic. PAI-sizer AppTest:
0 exceptions across the full rung × maturity matrix.)

## 10. Ownership

| Surface | Owns | Tag |
|---|---|---|
| ratchet | capability dict + precision-set param + tests | [ratchet] |
| PAI sizer | UI radio + maturity toggle + compare view | [pai-sizer] |
| Skippy | this spec + benefit narrative (deck/whitepaper) | [docs] |

Discipline (per CLAUDE.md): Skippy authors the cross-surface spec; ratchet
implements the capability; pai-sizer consumes. One-way: docs → ratchet → sizer.
