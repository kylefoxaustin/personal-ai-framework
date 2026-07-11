# Skippy on the Jetson AGX Orin

Scripts for running Skippy's LLM corpus on the shared `orin-agx` board.

## Before you touch the board

```bash
/res-status orin-agx        # who holds it
/res-request orin-agx       # join the FIFO queue; you get pinged + a 15m grace window
/reserve  orin-agx 2h hard  # claim it
/keep     orin-agx 90m      # heartbeat — non-GPU resources have no idle detection
/release  orin-agx          # as soon as you're done; someone is usually waiting
```

**Do not generate load on a board you do not hold.** SSH transfers burn board CPU
on decrypt and touch the memory bus; they will corrupt someone else's timing run.

## Use the LAN address, not the `orin` alias

`~/.ssh/config` maps `Host orin` → `192.168.55.1`, the **USB device-mode gadget**.
Measured 2026-07-09:

| path | address | throughput |
|---|---|--:|
| USB gadget (`ssh orin`) | 192.168.55.1 | 23–35 MB/s |
| LAN, board `eno1` @ 1 Gb/s | **10.0.1.124** | **77 MB/s** |

Two concurrent rsyncs sustain ~52 MiB/s aggregate. Staging 42 GiB takes ~15 min.

## Board environment

`torch` is installed but **fails to import** with `libcusparseLt.so.0` missing.
The library is present, just not on the loader path — no install required:

```bash
CUSP=$(dirname $(find ~/.local/lib -name 'libcusparseLt.so.0' | head -1))
export LD_LIBRARY_PATH="$CUSP:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
```

Present: `torch 2.5.0a0(nv24.08)`, `transformers 5.13.0`, `tensorrt 10.3.0`, `trtexec`.
Absent: `llama.cpp`, `vllm`, `accelerate`, `compressed-tensors`.
Compute capability **8.7**. `torch._int_mm` (INT8 tensor-core matmul) **works** —
in contrast to the 5090's SM120, which refuses vLLM's `cutlass_scaled_mm` INT8.

## The scripts

| file | what it does | needs |
|---|---|---|
| `orin_hf_perf.py` | decode rate, TTFT, power, achieved GB/s for HF fp16 models | torch + transformers (preinstalled) |
| `stage.sh` | rsync the GGUF corpus over the **LAN** | — |
| `build_llamacpp.sh` | build llama.cpp CUDA for SM87 | ⚠ needs authorization (clones external repo) |
| `run_orin.py` | full Q4/Q8/MoE ladder via `llama-server` | llama.cpp |

### `orin_hf_perf.py` — the weight-streaming law

qualcomm's `IQ9075_BOARD_DOSSIER.md` §4.4 *derives* LLM decode latency as
`weight_bytes ÷ memory_bandwidth`. That premise underpins the whole §4.4/§4.6
latency table, and it has only ever been checked against one datapoint on one bus.

Two models, same architecture (`Qwen2ForCausalLM`), different weight footprints:

```
  skippy-7b-v4   14.19 GiB fp16      ratio 1 : 1.94
  qwen2.5-14b    27.51 GiB fp16
```

If decode is purely bus-bound, `tok/s(14B) / tok/s(7B) = 1/1.94 = 0.516`.

**Prefill is separated from decode** by timing the same prompt twice — once for
1 new token, once for N — so prefill cancels exactly:

```
  decode_tok_s = (N - 1) / (t_N - t_1)
  ttft_ms      = t_1 * 1000
```

Run with the **no-RAG** pack (contexts 42–168 chars). The RAG pack's ~24K-char
contexts add KV-cache traffic and attention cost to decode, which is not weight
streaming and would confound the measurement.

```bash
python3 orin_hf_perf.py \
  --models skippy-7b-v4=~/skippy_corpus/models/skippy-7b-v4-fp16 \
           qwen2.5-14b=~/skippy_corpus/models/qwen2.5-14b-fp16 \
  --pack ~/skippy_corpus/out/pack_norag.jsonl \
  --out-dir ~/skippy_corpus/out --n-prompts 6 --n-new 128
```

It also reports achieved bandwidth (`weight_bytes × tok/s`) against the board's
204.8 GB/s spec. **If utilization comes back above 100%, the weight-streaming
assumption is false** — weights are partly cache-resident and not re-streamed per
token. That would be a result, not a bug.
