# Private NPU + CNN anchor secrets — spec for PAI sizer (and keyhole-sizer)

**Date:** 2026-05-14
**Status:** Spec / reusable template. PAI sizer vendors first; keyhole-sizer mirrors after.
**Why this exists:** Real NPU silicon measurements (NPU Mid INT8, NPU High FP/INT8) and CNN benchmarks are sensitive — they should never appear in chat transcripts, git history, or Drive. Streamlit secrets keeps them encrypted at rest, injected at runtime, with the local-vs-cloud workflow described below.

## Workflow at a glance

1. Numbers live in `.streamlit/secrets.toml` (gitignored). Author them on your local machine; never paste raw values into chat.
2. For local Streamlit runs, the file is read automatically — `st.secrets[...]` works out of the box.
3. For deploys to share.streamlit.io: copy the local TOML contents → paste into the app's **Settings → Secrets** textbox → save. Streamlit encrypts and injects at runtime.
4. Updates: edit local TOML and re-paste into Cloud Secrets when numbers change.

Discipline rule: when discussing measurements over bus or with Claude, refer to KEYS (`npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe.tokps`) not VALUES. The schema and loader can be authored in chat; the numbers never need to be.

## TOML schema — `.streamlit/secrets.toml`

```toml
# PAI sizer secrets — gitignored. Numbers are placeholder zeros here; real values
# live only in your local .streamlit/secrets.toml and in Streamlit Cloud Secrets.

PASSWORD = "..."  # existing auth secret


# ─────────────────────────────────────────────────────────────────────
# LLM anchor measurements — 3 NPU tier-precision cells × 3 models
# Bandwidth derivation inputs stored alongside the measurement so the
# sizer can re-derive implied bytes-per-token under different share
# assumptions (e.g., when the 100/75/50/25% share-selector overrides).
# ─────────────────────────────────────────────────────────────────────

#  Tier-precision cells:
#    mid_int8   = NPU Mid (INT8-only)
#    high_int8  = NPU High at INT8 (400 eTOPS)
#    high_fp    = NPU High at FP  (200 eTOPS)
#
#  Model keys:
#    qwen3_30b_a3b_moe   — Qwen3 30B-A3B Instruct (MoE, 3B active)
#    qwen25_32b_dense    — Qwen 2.5 32B Instruct (dense)
#    qwen25_7b_dense     — Qwen 2.5 7B Instruct (dense)

[npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe]
tokps                 = 0.0           # measured decode tok/s — REAL value goes here
prefill_tokps         = 0.0           # optional; set 0 if not measured
mem_gb                = 0.0           # runtime memory footprint
seqlen                = 2048          # measurement context length
source                = "measured"    # measured | vendor_spec | projected
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4         # 8.4 GT/s × 128-bit LPDDR5 ÷ 8 = 134.4 GB/s
bw_share_frac         = 0.75          # default share; UI may override
bw_efficiency_frac    = 0.70          # matches keyhole BW-efficiency methodology
notes                 = ""

[npu_llm_anchors.mid_int8.qwen25_32b_dense]
tokps                 = 0.0
prefill_tokps         = 0.0
mem_gb                = 0.0
seqlen                = 2048
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

[npu_llm_anchors.mid_int8.qwen25_7b_dense]
tokps                 = 0.0
prefill_tokps         = 0.0
mem_gb                = 0.0
seqlen                = 2048
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

# NPU High INT8 — likely different peak_bw_gbps; fill in actual silicon spec.
# Typical guess: LPDDR5X at 8.5+ GT/s × 256b ≈ 270+ GB/s. Confirm with silicon datasheet.

[npu_llm_anchors.high_int8.qwen3_30b_a3b_moe]
tokps                 = 0.0
prefill_tokps         = 0.0
mem_gb                = 0.0
seqlen                = 2048
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 0.0           # CONFIRM from silicon spec
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

[npu_llm_anchors.high_int8.qwen25_32b_dense]
tokps                 = 0.0
# ...same fields as above

[npu_llm_anchors.high_int8.qwen25_7b_dense]
tokps                 = 0.0
# ...same fields

# NPU High FP — same silicon, FP path (200 eTOPS).
# peak_bw_gbps is the SAME bus as INT8 (same physical DRAM); efficiency may differ.

[npu_llm_anchors.high_fp.qwen3_30b_a3b_moe]
tokps                 = 0.0
prefill_tokps         = 0.0
mem_gb                = 0.0
seqlen                = 2048
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 0.0           # same value as high_int8
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

[npu_llm_anchors.high_fp.qwen25_32b_dense]
tokps                 = 0.0
# ...

[npu_llm_anchors.high_fp.qwen25_7b_dense]
tokps                 = 0.0
# ...


# ─────────────────────────────────────────────────────────────────────
# CNN anchor measurements — 2-3 tier-precision cells × 3 CNN variants
# ─────────────────────────────────────────────────────────────────────

#  CNN keys:
#    resnet50      — ResNet-50 (confirm variant; ResNet-18 if different)
#    yolov8n_w4    — YOLOv8n with 4-bit weights
#    yolov8n_w8    — YOLOv8n with 8-bit weights

[cnn_anchors.mid_int8.resnet50]
ms_per_inference      = 0.0           # measured latency, milliseconds
fps                   = 0.0           # 1000 / ms; precompute or compute live
mem_mb                = 0.0           # runtime memory footprint
input_res             = "224x224"     # standard ImageNet ResNet-50 input
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

[cnn_anchors.mid_int8.yolov8n_w4]
ms_per_inference      = 0.0
fps                   = 0.0
mem_mb                = 0.0
input_res             = "640x640"     # confirm; typical YOLOv8 default
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

[cnn_anchors.mid_int8.yolov8n_w8]
ms_per_inference      = 0.0
fps                   = 0.0
mem_mb                = 0.0
input_res             = "640x640"
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

[cnn_anchors.high_int8.resnet50]
ms_per_inference      = 0.0
# ...same fields as cnn_anchors.mid_int8.resnet50 with peak_bw_gbps reflecting High silicon

[cnn_anchors.high_int8.yolov8n_w4]
# ...

[cnn_anchors.high_int8.yolov8n_w8]
# ...

# Optional: high_fp variants if CNN was measured under FP path on NPU High.
# Skip these sections if you only have INT8 numbers for CNN.
# [cnn_anchors.high_fp.resnet50] ...
```

## Loader module — `sizer/npu_anchors.py`

```python
"""Private NPU + CNN anchor loader.

Numbers live in Streamlit secrets (.streamlit/secrets.toml locally; Cloud
Secrets in production). This module exposes typed accessors with graceful
fallback when secrets aren't set (returns None → app falls back to
projection or shows 'not measured').

Bandwidth derivation: stored peak_bw_gbps × bw_share_frac × bw_efficiency_frac
gives the achieved bandwidth used to back out bytes-per-token. The
share_frac is overridable at call time so the UI's 100/75/50/25%
share-selector can re-derive on the fly without re-reading secrets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


# Badge color by source — matches keyhole-sizer's _render_source_banner convention.
BADGE_FOR_SOURCE = {
    "measured":     "🟢",
    "vendor_spec":  "🟡",
    "projected":    "🟠",
    # unknown / missing source → no badge
}


@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    prefill_tokps: float
    mem_gb: float
    seqlen: int
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override: Optional[float] = None) -> float:
        """BW available to NPU, applying any UI share override."""
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac

    def bytes_per_token(self, share_override: Optional[float] = None) -> float:
        """Memory bytes moved per decoded token (BW-bound decode model)."""
        if self.tokps <= 0:
            return 0.0
        return self.achieved_bw_gbps(share_override) * 1e9 / self.tokps


@dataclass(frozen=True)
class CNNAnchor:
    ms_per_inference: float
    fps: float
    mem_mb: float
    input_res: str
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override: Optional[float] = None) -> float:
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac


def _try_get(section: str, sub: str, key: str) -> Optional[dict]:
    """Defensive .get-chain for st.secrets — returns None on any miss."""
    try:
        return dict(st.secrets[section][sub][key])
    except Exception:
        return None


def load_llm_anchor(tier: str, precision: str, model_key: str) -> Optional[LLMAnchor]:
    """tier in {'mid','high'}, precision in {'int8','fp'}, model_key e.g. 'qwen3_30b_a3b_moe'.

    Returns None if the entry isn't in secrets — caller falls back to projection.
    """
    sub = f"{tier}_{precision}"
    data = _try_get("npu_llm_anchors", sub, model_key)
    if data is None or data.get("tokps", 0) <= 0:
        return None
    return LLMAnchor(
        tokps=float(data["tokps"]),
        prefill_tokps=float(data.get("prefill_tokps", 0.0)),
        mem_gb=float(data.get("mem_gb", 0.0)),
        seqlen=int(data.get("seqlen", 0)),
        source=str(data.get("source", "")),
        measured_date=str(data.get("measured_date", "")),
        peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
        bw_share_frac=float(data.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(data.get("bw_efficiency_frac", 0.70)),
        notes=str(data.get("notes", "")),
    )


def load_cnn_anchor(tier: str, precision: str, cnn_key: str) -> Optional[CNNAnchor]:
    """tier in {'mid','high'}, precision in {'int8','fp'}, cnn_key e.g. 'resnet50'."""
    sub = f"{tier}_{precision}"
    data = _try_get("cnn_anchors", sub, cnn_key)
    if data is None or data.get("ms_per_inference", 0) <= 0:
        return None
    fps = float(data.get("fps", 0.0))
    if fps <= 0 and data.get("ms_per_inference", 0) > 0:
        fps = 1000.0 / float(data["ms_per_inference"])
    return CNNAnchor(
        ms_per_inference=float(data["ms_per_inference"]),
        fps=fps,
        mem_mb=float(data.get("mem_mb", 0.0)),
        input_res=str(data.get("input_res", "")),
        source=str(data.get("source", "")),
        measured_date=str(data.get("measured_date", "")),
        peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
        bw_share_frac=float(data.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(data.get("bw_efficiency_frac", 0.70)),
        notes=str(data.get("notes", "")),
    )
```

## App.py integration snippet

```python
import streamlit as st
from sizer.npu_anchors import load_llm_anchor, load_cnn_anchor

# Existing share-selector knob (per project_npu_share_selector roadmap)
share_pct = st.sidebar.select_slider(
    "BW share available to NPU",
    options=[0.25, 0.50, 0.75, 1.00],
    value=0.75,
    format_func=lambda x: f"{int(x*100)}%",
)

st.subheader("LLM throughput — NPU Mid INT8")
for model_label, key in [
    ("Qwen3 30B-A3B MoE (3B active)", "qwen3_30b_a3b_moe"),
    ("Qwen 2.5 32B dense",            "qwen25_32b_dense"),
    ("Qwen 2.5 7B dense",             "qwen25_7b_dense"),
]:
    anchor = load_llm_anchor("mid", "int8", key)
    if anchor is None:
        st.metric(f"⏸ {model_label}", "not measured")
        continue
    bytes_per_tok = anchor.bytes_per_token(share_override=share_pct)
    st.metric(
        f"{anchor.badge} {model_label}",
        f"{anchor.tokps:.1f} tok/s",
        delta=f"{bytes_per_tok/1e6:.0f} MB/tok at {int(share_pct*100)}% BW share",
        delta_color="off",
    )

# CNN section follows same pattern with load_cnn_anchor
```

## Gitignore + Cloud-deploy flow

1. **`.gitignore`** must include:
   ```
   .streamlit/secrets.toml
   ```
   (PAI sizer's existing gitignore already has this line — confirmed 2026-05-14.)

2. **Local dev:** populate `.streamlit/secrets.toml` (use the schema above as the template — replace zeros with measured values). `streamlit run app.py` reads it automatically.

3. **Streamlit Cloud deploy:**
   - share.streamlit.io → app → Settings → Secrets
   - Copy entire contents of local `secrets.toml` → paste into the multi-line textbox
   - Save. Streamlit encrypts and injects at runtime.
   - App code (`st.secrets[...]`) works identically in both environments.

4. **Updating numbers:**
   - Edit local TOML → reruns pick it up immediately
   - For Cloud: re-paste updated TOML into Secrets tab; save; redeploy auto-triggers

## Cross-app coordination

When [pai-sizer] vendors this:
- Drop `sizer/npu_anchors.py` in PAI sizer repo
- Append the new sections to `.streamlit/secrets.toml.example` (keeping zero placeholders)
- Kyle's `.streamlit/secrets.toml` (local) gets real numbers; never committed
- App.py integrates via the snippet pattern above
- Streamlit Cloud Secrets tab gets the TOML contents pasted

When [sizer] (keyhole-sizer) mirrors:
- Same schema (it's NPU-tier-agnostic; the data shape transfers cleanly)
- Same loader (`sizer/npu_anchors.py` mirrored)
- Different `peak_bw_gbps` per tier if keyhole models different silicon — same toml shape, different numbers in each side's secrets

## Discipline reminders

- Never paste raw measurement values into chat — refer by key name (`npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe.tokps`).
- Never commit `.streamlit/secrets.toml`. If you accidentally do, immediately rewrite history + rotate the assumption (a public commit of these numbers is a leak).
- `secrets.toml.example` (with placeholder zeros) IS safe to commit — that's the schema reference for collaborators.
- BW-share-selector UI overrides the `bw_share_frac` field at call time. Default in secrets is 0.75; UI default matches.
- Bus messaging across sessions: discuss SHAPES (`schema`, `axes`, `field names`) not VALUES.

## Outstanding questions for Kyle (to fill in once schema lands)

- **NPU High peak_bw_gbps**: which LPDDR rate + bus width? (LPDDR5X 8.5+ GT/s × 256b ≈ 270 GB/s is the guess; confirm from datasheet.)
- **ResNet variant**: ResNet-50 (224×224) assumed; correct?
- **YOLOv8n input resolution**: 640×640 assumed; correct?
- **CNN High FP variant**: did you measure CNN under FP path on NPU High, or INT8-only? (Affects whether `cnn_anchors.high_fp.*` sections are needed.)
