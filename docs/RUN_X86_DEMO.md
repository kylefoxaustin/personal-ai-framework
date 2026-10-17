# Run the Agentic-Edge Demo on Your Own x86 Laptop (pure CPU)

**Goal: clone the repo, run one command, and reproduce the headline finding on your own
machine — no GPU, no Docker, no server, no fleet.** It downloads a small model, runs a
CPU decode benchmark, and shows you that

> **decode (token generation) is bounded by your DDR memory bandwidth, not by compute/TOPS.**

Works on **Linux** (native) and **Windows** (native or WSL2). Intel or AMD x86. A laptop is fine.

---

## 1. Install (one wheel)

You need Python 3.9+, `llama-cpp-python` (the same CPU runtime the project uses), and
`huggingface-hub` (used to download the model). No CUDA.

**Linux / WSL2:**
```bash
pip install llama-cpp-python huggingface-hub
```

**Windows (native, PowerShell):** use the prebuilt CPU wheel so no compiler is needed:
```powershell
py -m pip install --prefer-binary llama-cpp-python huggingface-hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
`--prefer-binary` matters. Without it, pip may pick a newer source release from PyPI and try to
compile it. That needs the Visual Studio "Desktop development with C++" Build Tools and CMake,
and it fails with a long CMake error when they are missing.

Get Python from python.org (tick "Add python.exe to PATH") or `winget install Python.Python.3.12`.
On Windows, `python3` often opens the Microsoft Store stub. Use `py` or `python` instead.

> **Windows tip:** WSL2 (Ubuntu) is the smoothest path and matches the Linux instructions exactly.
> Native Windows works too via the prebuilt wheel.

## 2. Run

```bash
python3 scripts/x86_demo.py                 # Linux/WSL2. Default: Qwen2.5-3B (2 GB, laptop-friendly)
```
```powershell
py scripts\x86_demo.py                      # Windows (PowerShell or cmd), same flags
```

That's it. The script self-downloads the GGUF from Hugging Face on first run (cached afterward),
checks it fits in your RAM, and benchmarks prefill + decode on your CPU.

Options:
```bash
python3 scripts/x86_demo.py --model 7b        # the ladder's reference size (needs ~6 GB free RAM)
python3 scripts/x86_demo.py --threads 8        # override thread count (default: physical-core-ish)
python3 scripts/x86_demo.py --ddr-gbps 89.6    # supply your peak DDR bandwidth for the util% headline
python3 scripts/x86_demo.py --tasks            # also run 3 real agentic tasks (see below)
```

### `--tasks` — run real agentic work, model only

The default run is a raw decode/prefill micro-benchmark (the bandwidth headline). `--tasks` adds
three **real** agentic tasks that need **nothing but the loaded model** — no server, no ChromaDB,
no network, no extra pip installs:

1. **Email drafting** — pure generation.
2. **File read + summarize** — reads a bundled local notes file → 3 bullets (a file-tool + decode).
3. **Grounded spec Q&A (inline context)** — the datasheet facts are pasted into the prompt instead
   of retrieved, so it's grounded *without* a vector DB ("poor-man's RAG").

Each prints its output (read it — the coherence gate) plus a **DERIVED decode-share**: the
prefill/decode time split estimated from the measured rates. You'll see every task is
**70–90%+ decode** — i.e. even real agentic prompts are bandwidth-bound, which is the whole point.

> **What you'll notice on the 3B:** the grounded Q&A is accurate (the facts are right there), the
> email is coherent, and the file-summary may **fabricate** details not in the notes. That's not a
> demo glitch — it's the finding: a small model's accuracy is bounded by *grounding*, not size.
> `--model 7b` grounds noticeably better. The other 7 tasks of the full benchmark (RAG, ASR, OCR,
> web, tool-chains) need infrastructure and live in `docs/RUN_BENCHMARKS.md`.

## 3. Read the result

Example (a real run on an Intel i9-14900KF desktop, dual-channel DDR5-5600):
```
=== x86/AMD pure-CPU decode-ladder demo ===
  CPU      : Intel(R) Core(TM) i9-14900KF  (32 logical cores, using 16 threads)
  SIMD     : AVX2, AVX, FMA, F16C  (llama.cpp build — prefill speed depends on this)
  RAM      : 101.0 GB
  DDR peak : 89.6 GB/s (user-supplied)  [SOURCED]
  model    : qwen2.5-3b  (Q4_K_M, 2.0 GB weights, 3.1B params)
  gate     : RUNNABLE — fits: needs ~3.5 GB, box has 101.0 GB

  --- result ---
  prefill        :    228.4 tok/s   [MEASURED]  (compute-bound; scales with --threads)
  decode         :    32.95 tok/s   [MEASURED]  (bandwidth-bound; ~flat vs threads)
  eff. mem BW    :     65.9 GB/s    [DERIVED: decode x weights]
  DDR util       :     73.5 %       [DERIVED/SOURCED] <- decode runs at 73.5% of peak DDR BW

  ==> Decode is memory-bandwidth-bound: the generation loop sustains 73.5% of this box's
      peak DDR bandwidth with ZERO GPU/NPU. More TOPS would not move this number; more
      memory bandwidth (faster/more DIMM channels) would.
```
Your numbers will differ — that's the point; report your own CPU + DDR + date.

**What the numbers mean**
- **decode tok/s** — how fast it generates. This is what you feel in an interactive agent.
- **eff. mem BW** = `decode_tok/s x model_weight_bytes`. A dense model reads ~all its weights
  from DRAM once per generated token, so this is the memory bandwidth the decode loop actually
  pulls. It's a **DERIVED** number (labelled as such).
- **DDR util %** — that effective bandwidth as a fraction of your DIMMs' peak. On most laptops
  decode lands at **60–90% of peak DDR**, which is the finding: *the generation loop is fed by
  memory bandwidth, and an accelerator's TOPS are idle during it.*

### Provenance (why the labels)
Per the project's Fleet Law 1, every number carries a tag:
- `decode` / `prefill` tok/s → **MEASURED** (ran on your box).
- `eff. mem BW` → **DERIVED** (decode × weights).
- `DDR peak` → **SOURCED** (your DIMM speed, auto-detected or `--ddr-gbps`).

The honest headline is the **within-box util %** — a derived/sourced ratio of your own machine.
Don't compare your laptop's derived effective-BW bare against another board's measured tok/s;
that's a mixed-tier comparison. Compare decode tok/s to decode tok/s.

## 4. (Optional) see it in the cross-board ladder

```bash
python3 scripts/aggregate_boards.py
```
Your `localhost-x86` rung is written to `eval/results/ladder/board_runs/localhost-x86.json` and
joins the ladder next to the Jetsons / i.MX95 rungs (where those runs exist).

---

## Scope

This demo is the **decode-ladder tier** — the part that reproduces the bandwidth finding with the
least setup. It does **not** run the full 10-task agentic harness (that needs the Skippy server +
ChromaDB + RAG corpus; see `docs/RUN_BENCHMARKS.md`). The full harness on x86 is a later tier; the
decode ladder is the thing a stranger can run on a laptop in two commands, which is the point here.

## Troubleshooting
- **`missing package: llama_cpp` / `huggingface_hub`** — see step 1. On Windows use the
  prebuilt-wheel index with `--prefer-binary`.
- **Windows: `DLL load failed` importing llama_cpp** — install the Microsoft Visual C++
  Redistributable (x64), then reopen the terminal.
- **Windows: symlink warning from huggingface_hub** — harmless. The cache copies the file instead
  of symlinking it. Enable Developer Mode to silence it.
- **`UNRUNNABLE — won't fit`** — the model is bigger than your free RAM. Use `--model 3b` (default)
  or close other apps. The gate refuses rather than thrash swap.
- **`DDR peak: unknown`** — auto-detect needs `sudo dmidecode` (Linux) or PowerShell's
  `Get-CimInstance Win32_PhysicalMemory` (Windows; falls back to `wmic` on old builds). It counts
  DIMMs as channels, so a 4-DIMM dual-channel desktop reads 2x too high, and soldered LPDDR laptops
  can misreport. Pass
  `--ddr-gbps <peak>` to get the util% (peak = channels × MT/s × 8 ÷ 1000; e.g. dual-channel
  DDR5-5600 = 2×5600×8÷1000 ≈ 89.6 GB/s).
- **Download is slow / offline** — the GGUF comes from Hugging Face on first run and is cached. On
  a metered/offline box, pre-download it once on a connected machine (same HF repo in
  `eval/model_registry.json`).
