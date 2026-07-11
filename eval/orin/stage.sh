#!/usr/bin/env bash
# Stage the Skippy precision-ladder corpus onto the Jetson AGX Orin.
#
# CRITICAL: use the LAN address, NOT the `orin` ssh alias.
#   `orin` → 192.168.55.1  = USB device-mode gadget  →  23-35 MB/s
#   10.0.1.124             = eno1, gigabit ethernet  →  77 MB/s
# Measured 2026-07-09. At 49 GB that is ~40 min vs ~11 min.
#
# Run from the repo root. Requires the orin lease to be YOURS (`/res-status orin`).
set -euo pipefail

ORIN="${ORIN:-kyle@10.0.1.124}"
DEST="${DEST:-/home/kyle/skippy_corpus}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODELS=(
  "models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf"
  "models/kyle-7b-v4-q4_k_m.gguf"
  "models/qwen2.5-14b-hf/qwen2.5-14b-instruct-stock-q4_k_m.gguf"
  "models/qwen2.5-14b-q8/Qwen2.5-14B-Instruct-Q8_0.gguf"
  "models/qwen3-30b-a3b-instruct-2507/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
)

echo "▶ target: $ORIN:$DEST"
ssh "$ORIN" "mkdir -p $DEST/models $DEST/bundle $DEST/out"

echo "▶ free space on target:"
ssh "$ORIN" "df -h $DEST | tail -1"

echo "▶ staging bundle (1.1 MB)"
rsync -az --info=progress2 "$REPO/eval/portable_bundle/" "$ORIN:$DEST/bundle/"
rsync -az "$REPO/eval/orin/run_orin.py" "$ORIN:$DEST/"

echo "▶ staging models (~49 GB) — flat into $DEST/models/"
for m in "${MODELS[@]}"; do
  [ -f "$REPO/$m" ] || { echo "  ! missing $m" >&2; exit 1; }
  echo "  → $(basename "$m")"
  # --partial + --append-verify so an interrupted transfer resumes rather than restarts.
  rsync -a --partial --append-verify --info=progress2 \
        "$REPO/$m" "$ORIN:$DEST/models/"
done

echo "▶ verifying sizes match"
for m in "${MODELS[@]}"; do
  b="$(basename "$m")"
  local_sz=$(stat -c%s "$REPO/$m")
  remote_sz=$(ssh "$ORIN" "stat -c%s $DEST/models/$b" 2>/dev/null || echo 0)
  if [ "$local_sz" != "$remote_sz" ]; then
    echo "  ✗ SIZE MISMATCH $b: local=$local_sz remote=$remote_sz" >&2
    exit 1
  fi
  echo "  ✓ $b"
done
echo "✓ staged."
