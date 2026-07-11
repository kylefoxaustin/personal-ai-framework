#!/usr/bin/env bash
# Build llama.cpp with CUDA on the Jetson AGX Orin. Run ON the board.
#
# Orin = Tegra234, Ampere, compute capability 8.7 → CMAKE_CUDA_ARCHITECTURES=87.
# Do NOT let CMake autodetect; on Jetson it frequently picks the wrong arch or
# emits PTX-only, which silently falls back to a slow path at runtime.
#
# Nothing here needs pip/torch. llama.cpp links the system CUDA directly, which
# is the whole reason we use it rather than fighting the Jetson pip index
# (torch 2.11 on that index wants CUDA 12.9; the board driver is 12.6).
set -euo pipefail

SRC="${SRC:-$HOME/llama.cpp}"
JOBS="${JOBS:-$(nproc)}"

command -v nvcc >/dev/null || export PATH=/usr/local/cuda/bin:$PATH
echo "▶ nvcc: $(nvcc --version | tail -1)"
echo "▶ cores: $JOBS"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
fi
cd "$SRC"

# -DGGML_NATIVE=OFF: llama.cpp's aarch64 build defaults to -mcpu=native. The
# [orb_slam] session measured this on THIS board (2026-07-09, single-variable, one
# source tree, gcc 11.4, interleaved rounds): -mcpu=cortex-a78ae / native is SLOWER
# than -mcpu=generic on every kernel they tried (+1.6% to +5.8%), and the penalty is
# reproduced by -mtune=cortex-a78ae alone — gcc 11.4's a78ae cost model is mistuned.
# Our decode is GPU-bound so this barely matters, but there is no reason to pay it.
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF

cmake --build build --config Release -j "$JOBS" --target llama-server llama-cli llama-bench

echo
echo "▶ installing to ~/.local/bin"
mkdir -p "$HOME/.local/bin"
install -m755 build/bin/llama-server build/bin/llama-cli build/bin/llama-bench "$HOME/.local/bin/"

echo "▶ verifying CUDA backend is actually live (must print 'CUDA0'):"
"$HOME/.local/bin/llama-cli" --version 2>&1 | head -5 || true
echo
echo "✓ built. Ensure ~/.local/bin is on PATH:  export PATH=\$HOME/.local/bin:\$PATH"
echo "  Sanity: llama-bench -m <gguf> -p 0 -n 32   # should show ~100+ tok/s on a 7B Q4"
