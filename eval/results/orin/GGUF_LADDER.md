# Orin GGUF precision ladder + the MoE decode law

**Date:** 2026-07-09 · **Board:** `orin-agx` (Tegra234, sm_87, 204.8 GB/s LPDDR5, MAXN)
**Runtime:** llama.cpp CUDA (`082b326`), `-DGGML_NATIVE=OFF`, `llama-bench -p 0 -n 128 -r 3`
Raw: `ladder.log`, `tegra_*.log` in this directory.

---

## Result

| config | streamed/token | tok/s | achieved | % of bus | watts | tok/s/W |
|---|--:|--:|--:|--:|--:|--:|
| fp16 dense 7B | 14.19 GiB | 10.76 | 163.9 GB/s | 80.1% | 27.07 | 0.40 |
| fp16 dense 14B | 27.51 GiB | 5.56 | 164.2 GB/s | 80.2% | 26.74 | 0.21 |
| Q8_0 dense 14B | 14.62 GiB | 10.79 | 169.4 GB/s | **82.7%** | 31.60 | 0.34 |
| Q4_K_M dense 7B | 4.36 GiB | 26.50 | 124.1 GB/s | 60.6% | 32.83 | 0.81 |
| Q4_K_M dense 14B | 8.37 GiB | 13.76 | 123.7 GB/s | 60.4% | 32.46 | 0.42 |
| **Q4_K_M MoE 30B-A3B** | **1.95 GiB** | **43.74** | 91.6 GB/s | 44.7% | 27.07 | **1.62** |

`skippy-7b-v4` (production fine-tune, Q4_K_M, 4.07 GiB): **27.82 tok/s**, 121.6 GB/s.

## 1. The MoE decode law — pre-registered, then measured

Predictions written down **before** the run (`RESULTS.md`), using the Q4-measured
achieved bandwidth of 123.7 GB/s:

| hypothesis | predicted | outcome |
|---|--:|---|
| decode streams **all** 17.28 GiB | 6.7 tok/s | **refuted by 6.6×** |
| streams **active only**, gather free | 61.6 tok/s | too high |
| streams active only, **gather penalty** (from qualcomm's iq9) | ~31.6 tok/s | too low |
| **measured** | **43.74 ± 0.68** | active-only, with a ~26% gather penalty |

**Decode tracks active parameters. Confirmed.** The 30B MoE decodes **3.18× faster
than the dense 14B** (43.74 vs 13.76 tok/s) while being **2.06× larger on disk**.

Computing "achieved bandwidth" from the MoE's *total* file size yields **396% of the
bus** — physically impossible, and the cleanest possible proof that decode does not
stream all weights.

### Exact per-token bytes, from the GGUF tensor table

Not estimated as `total × 3.3/30.5`. Read from the file, sizing each tensor by
**offset deltas** (authoritative — a hand-written ggml type table got Q6_K wrong and
under-counted by 3 GiB):

```
  expert tensors (*_exps)  16.348 GiB   (144 tensors, 128 experts, 8 used/token)
  non-expert                0.929 GiB
  per token = 0.929 + (8/128) × 16.348 = 1.951 GiB
```

## 2. Achieved bandwidth is SIZE-invariant but PRECISION-dependent

Three dense Q4 models, 2× apart in size, all land on **121.6–124.1 GB/s**. The bus is
the invariant; tok/s is `bandwidth ÷ streamed_bytes`. But the *bandwidth* itself
depends on what has to happen to each byte:

```
  fp16 / Q8_0        80–83% of bus   pure weight streaming, nothing in the way
  Q4_K_M dense       ~60%            k-quant dequantization
  Q4_K_M MoE         ~45%            + scattered 8-of-128 expert gather
```

**Q4_K_M's speedup over Q8_0 is 1.27×, not the 1.75× its byte ratio predicts.**
Dequant eats 27% of the theoretical win.

**And Q4 draws MORE power than fp16** (32.5 W vs 27.1 W) while achieving *less*
bandwidth. Dequantization is compute, and you pay for it in both currencies. Anyone
choosing Q4 purely for "less memory traffic → less energy" has it backwards on this
part.

## 3. Two-board agreement on the MoE gather penalty

`[qualcomm]` measured the same model on the IQ-9075's Hexagon v73 at **15.63 tok/s**.
Applying the exact per-token figure above (1.951 GiB):

| board | bus spec | MoE achieved | % of bus |
|---|--:|--:|--:|
| IQ-9075 (Hexagon v73) | 76.8 GB/s | 32.7 GB/s | **42.6%** |
| Jetson AGX Orin | 204.8 GB/s | 91.6 GB/s | **44.7%** |

Two entirely different memory systems, **MoE bus utilization within two points.**
Dense on Orin reaches 60% (Q4) and 80% (fp16). The ~2× gap between dense and MoE looks
like a **property of the computation, not of the silicon**: gathering 8 of 128 experts
is a scattered read that defeats prefetch and coalescing, wherever you run it.

*(This assumes qualcomm ran Q4_K_M — asked, awaiting confirmation. If he ran Q8_0 the
comparison changes and this section needs revisiting.)*

## Caveats

- **MoE power (27.07 W) rests on 19 active tegrastats samples** — the model decodes
  384 tokens in ~8.8 s. The tok/s figure is solid (±0.68 over 3 reps); treat the watts
  as provisional.
- Power is the mean over samples where `VDD_GPU_SOC > 3× idle`. `tegrastats` ran across
  load *and* decode, and model load is disk-bound and near-idle on the GPU rail — a
  naive mean gives the MoE 4.18 W, which is nonsense (it loads 17 GiB).
- `llama-bench -p 0` measures decode only. No prefill, no RAG context.
- Built with `-DGGML_NATIVE=OFF` per `[orb_slam]`'s measurement on this board:
  `-mcpu=native`/`cortex-a78ae` is **slower** than generic (gcc 11.4's a78ae cost model
  is mistuned). Decode is GPU-bound so it barely matters — but there's no reason to pay it.
