# Jetson AGX Orin — LLM decode, power, and the weight-streaming law

**Date:** 2026-07-09 · **Board:** `orin-agx` (Tegra234, SM87, 204.8 GB/s LPDDR5, MAXN)
**Runtime:** torch 2.5.0a0 (nv24.08) + transformers 5.13.0, fp16, greedy, batch=1
**Script:** [`eval/orin/orin_hf_perf.py`](../../orin/orin_hf_perf.py) · raw logs in this directory

---

## Headline

**LLM decode on the Orin is bus-bound, and the weight-streaming law holds to 0.2%.**

| model | weights | tok/s | achieved | % of 204.8 GB/s | TTFT | power |
|---|--:|--:|--:|--:|--:|--:|
| `skippy-7b-v4` (production fine-tune) | 14.19 GiB | **10.76** | 163.9 GB/s | **80.0%** | 104.6 ms | **27.07 W** |
| `qwen2.5-14b-instruct` | 27.51 GiB | **5.558** | 164.2 GB/s | **80.2%** | 203.4 ms | **26.74 W** |

```
weight ratio                                   1 : 1.9394
predicted tok/s ratio if purely weight-streaming    0.5156
measured  tok/s ratio                               0.5166
ratio-of-ratios (1.000 = perfect bus-bound)         1.0020
```

**The ratio is not the strongest evidence — the achieved bandwidth is.** Two
models differing 2× in size land on **163.9** and **164.2 GB/s**. The bus is the
invariant; tok/s is simply `bandwidth ÷ weight_bytes`.

`TTFT` scaled **1.945×** against a weight ratio of 1.939 — at ~50-token contexts
even *prefill* is weight-bound. Prefill only becomes compute-bound once the
prompt is long enough for attention FLOPs to exceed the cost of reading the
weights once.

## Power (measured, `tegrastats`, no sudo required)

| state | VDD_GPU_SOC | VDD_CPU_CV | total |
|---|--:|--:|--:|
| idle | 2.66 W | 0.97 W | **3.2 W** |
| 7B decode | 25.62 W (peak 28.83) | 1.44 W | **27.07 W** |
| 14B decode | 25.12 W (peak 29.07) | 1.62 W | **26.74 W** |

Perf/W: 7B **0.397 tok/s/W**, 14B **0.208 tok/s/W**. Decode adds ~24 W over idle.

Both models draw within 1% of each other — they saturate the same bus at the same
80%, so they burn the same power. **The Orin's real sustained draw is ~27 W, not
the 60 W SoC ceiling** that perf/W comparisons had been assuming (2.2× high).

> **Scope.** This is *memory-bound LLM decode*. A batch-1 CNN that is
> launch-bound draws less. 27 W is an **upper bound** on the Orin's draw for
> `[backend]`'s vision corpus, not a substitute for measuring it.

## Why this matters beyond the Orin

`IQ9075_BOARD_DOSSIER.md` §4.4/§4.6 *derives* iq9 LLM/VLA latency as
`weight_bytes ÷ 76.8 GB/s` — dividing by the **full nameplate bus**. We measured
**80.1% of spec**, twice, on two model sizes. If the iq9's LPDDR5 behaves
similarly, its effective bus is ~61.5 GB/s and every derived number is optimistic
by **1.25×**:

| workload | dossier | @80% bus |
|---|--:|--:|
| 7B int8 backbone, per token | 91 ms | 114 ms |
| OpenVLA-7b (7 out tok) | 729 ms | 910 ms |
| **OpenDriveVLA-0.5B (20 tok)** | **134 ms (7.5 Hz)** | **167 ms (6.0 Hz)** |
| AutoVLA 3.7B (64 tok) | 3132 ms | 3909 ms |
| DriveVLM 3.8B (100 tok) | 4997 ms | 6239 ms |
| OpenEMMA 7.6B (128 tok) | 12766 ms | 15937 ms |

We do **not** claim 80% transfers to Hexagon — different controller, DRAM, and
access pattern. But **100% is not a defensible default**, and this is a measured
prior from a comparable weight-streaming decode on a comparable LPDDR5 part. An
on-board `icc_bwmon` read during a real decode would calibrate the whole table.

## Method note — separating prefill from decode

Naive `generate()` timing conflates prefill (compute-bound, scales with prompt
length) with decode (bus-bound, scales with weights). We time the same prompt
twice:

```
t1 = time to generate  1 new token  = prefill + 1 decode step
tN = time to generate  N new tokens = prefill + N decode steps

decode_tok_s = (N - 1) / (tN - t1)     # prefill cancels exactly
ttft_ms      = t1 * 1000
```

Verified against synthetic timings: recovers the true rate to machine precision.

Run on the **no-RAG** prompt pack (42–168 char contexts). The RAG pack's ~24K-char
contexts add KV-cache traffic and attention cost to decode, which is *not* weight
streaming and would confound the measurement.

## Environment gotchas (cost us time)

1. **`torch` won't import**: `libcusparseLt.so.0` missing. It is present, just not
   on the loader path — no install needed. Prepend
   `$(dirname $(find ~/.local/lib -name 'libcusparseLt.so.0'))` to `LD_LIBRARY_PATH`.
2. **transformers 5.13 passes `enable_gqa` to SDPA; JetPack's torch 2.5.0a0
   doesn't have it.** You cannot just drop the kwarg — both models use
   grouped-query attention (7B: 28 q / 4 kv; 14B: 40 / 8), and dropping it
   computes attention against mismatched head counts. `orin_hf_perf.py` installs
   a shim that expands k/v explicitly; verified bit-exact (`maxdiff = 0.0`)
   against the reference for both head configurations.
3. `tegrastats` needs **no sudo** and reports mW per rail as `NAME 1234mW/5678mW`
   (instant/avg).
4. **`torch._int_mm` (INT8 tensor-core matmul) works on SM87** — the integer path
   the 5090's SM120 refuses through CUTLASS. Hardware is there on both; only the
   kernel libraries differ.
5. **Don't route an INT8 measurement through TensorRT's implicit quantization**
   (`trtexec --int8`): per `[backend]`, it emits *reformat layers*, not int8
   kernels — `clip_vit` came out byte-identically as slow as fp16, i.e. nothing
   ran in int8. Use QDQ/explicit quantization, or llama.cpp/vLLM.

## Pre-registered predictions for the GGUF run (not yet executed)

The law says `tok/s = achieved_bandwidth ÷ weight_bytes`, with achieved bandwidth
measured at **164.05 GB/s** and empirically independent of model size. Writing
these down **before** running the GGUF ladder, so the MoE test is a test and not a
story told afterwards.

| model | weights | predicted tok/s |
|---|--:|--:|
| Qwen2.5-7B Q4_K_M | 4.36 GiB | **35.0** |
| Qwen2.5-7B Q8_0 | 8.10 GiB | 18.9 |
| Qwen2.5-14B Q4_K_M | 8.37 GiB | 18.3 |

Note the 7B fp16 row is *not* a prediction — it is the input from which the
achieved bandwidth was derived. The genuinely out-of-sample claims are the
quantized rows.

**Cross-check against our own deck.** `scripts/build_use_case_deck.py` projects
the Orin at *"Qwen 2.5 7B @ 25–40 tok/s"*, from a TOPS/bandwidth hand-wave. The
law predicts **35.0 tok/s** for 7B Q4_K_M — inside that band. So the deck's
projection is about to become measurement-backed rather than asserted.

### The MoE test — falsifiable, 9× spread

`Qwen3-30B-A3B-Instruct-2507` Q4_K_M is **17.28 GiB on disk** with **~3.3B active
params** (~1.9 GiB of expert+router weights touched per token).

| if decode streams… | predicted tok/s |
|---|--:|
| **all 17.28 GiB** (total params) | **8.8** |
| **~1.9 GiB** (active params only) | **80.4** |

A **9.1× spread.** The measurement cannot come back ambiguous. If it lands near
80, qualcomm's §4.5 prediction (*"decode latency tracks active params, footprint
tracks total"*) is confirmed and a 30B-class model decodes **3× faster than a
dense 14B** on this board. If it lands near 8.8, the "streams only active experts"
assumption is wrong on this runtime, and that is the more interesting result.

## Not done

- **Q4_K_M / Q8_0 rungs** and the **MoE decode measurement** — both need GGUF,
  hence `llama.cpp` built on the board (`eval/orin/build_llamacpp.sh`, authorized
  2026-07-09). The MoE run fills the gap qualcomm's own §4.5 flags: *"we did not
  run a dedicated MoE benchmark."*
- **W8A8 accuracy on real INT8 silicon.** `torch._int_mm` works on SM87, so the
  Orin can run the rung the 5090 refuses. Do **not** route it through TensorRT
  implicit quantization — that measures reformat overhead, not INT8.
- Corpus (42 GiB fp16) left staged at `~/skippy_corpus/` on the board.
