# Ecosystem Checkpoint — 2026-05-18

**Purpose:** Single consolidated stable-state document for the Skippy ecosystem (personal-ai-framework / personal-ai-assistant-sizer / keyhole / keyhole-sizer) at the v5.10.0 + v1.0.0 cross-repo checkpoint. Built for Claude Browser (and Kyle) before engine-extraction kickoff for the upcoming drone use case. Each surface's stable state is captured below in the same template form so the document can be merged section-by-section, or each surface can be read in isolation.

**Compiled:** 2026-05-18, from bus log entries posted between 17:53 and 17:56.
**Compiled by:** [docs] (personal-ai-framework session) at Kyle's request.

---

## Quick reference — all four surfaces

| Surface | Tag | Commit | Repo | Release URL | Live |
|---|---|---|---|---|---|
| personal-ai-framework (Skippy) | **v5.10.0** | `159ef88` | [github.com/kylefoxaustin/personal-ai-framework](https://github.com/kylefoxaustin/personal-ai-framework) | [v5.10.0 release](https://github.com/kylefoxaustin/personal-ai-framework/releases/tag/v5.10.0) | local FastAPI + web/index.html |
| personal-ai-assistant-sizer (PAI sizer) | **v1.0.0** | `7e34df0` | [github.com/kylefoxaustin/personal-ai-assistant-sizer](https://github.com/kylefoxaustin/personal-ai-assistant-sizer) | [v1.0.0 release](https://github.com/kylefoxaustin/personal-ai-assistant-sizer/releases/tag/v1.0.0) | [personal-ai-assistant-sizer.streamlit.app](https://personal-ai-assistant-sizer.streamlit.app/) (password-gated) |
| keyhole | **v1.0.0** | `a23af3b` | [github.com/kylefoxaustin/keyhole](https://github.com/kylefoxaustin/keyhole) | [v1.0.0 release](https://github.com/kylefoxaustin/keyhole/releases/tag/v1.0.0) | local FastAPI on `:8777` |
| keyhole-sizer | **v1.0.0** | `21debd2` | [github.com/kylefoxaustin/keyhole-sizer](https://github.com/kylefoxaustin/keyhole-sizer) | [v1.0.0 release](https://github.com/kylefoxaustin/keyhole-sizer/releases/tag/v1.0.0) | [keyhole-sizer.streamlit.app](https://keyhole-sizer.streamlit.app/) (password-gated) |

## Cross-cutting concerns (apply across all four surfaces)

### NPU canonical specs (locked 2026-05-14; PAI Slide 11 is golden)

| Tier | Bus | Peak BW | INT8 | FP |
|---|---|---|---|---|
| NPU Low | 64-bit LPDDR4 @ 4 GT/s | 32 GB/s | ~42 TOPS | — |
| NPU Mid | 128-bit LPDDR5X @ 8.4 GT/s | 134.4 GB/s | 200 TOPS | INT8 only |
| NPU High | 128-bit LPDDR5X @ 8.4 GT/s | 134.4 GB/s | 400 eTOPS | 200 eTOPS FP |

Mid + High share the same memory bus → identical decode TPS (37.85 on Qwen 3 30B-A3B). High wins on TTFT (2× faster, compute-bound) and on CNN/FP workloads. Tier difference is compute-only, not bandwidth.

### Private-anchor-secrets spec (canonical)

- **Spec:** `personal-ai-framework/docs/private_anchor_secrets_spec.md` (commit `65bf89c`)
- **9 LLM cells:** `npu_llm_anchors.{mid_int8|high_int8|high_fp}.{qwen3_30b_a3b_moe|qwen25_32b_dense|qwen25_7b_dense}`
- **6 CNN cells:** `cnn_anchors.{mid_int8|high_int8}.{resnet50_w4|yolov8n_w4|yolov8n_w8}`
- **Per-cell fields:** `tokps`/`ms_per_inference` + `peak_bw_gbps` + `bw_share_frac` + `bw_efficiency_frac` + `source` + `measured_date`
- **Storage:** Gitignored `.streamlit/secrets.toml` locally; pasted into Streamlit Cloud Secrets at deploy time (encrypted at rest). Consumers: PAI sizer + keyhole-sizer + keyhole backend (`--include-private` flag).

### Reciprocal cross-references in decks

| From | To | Mechanism |
|---|---|---|
| Skippy deck Slide 18 | Keyhole deck | "What Keyhole is" two-column slide; points at `gdrive:skippy_files/keyhole/look_here/` |
| Keyhole deck Slide 48 | Skippy deck | "Optional LLM layer — the Skippy product artifact, unmodified"; references methodology canon in PAI deck |

### Discipline rules (in force across all surfaces)

- **KEY-not-VALUE** for anchor-secrets data (treat like credentials; no measurement values in source / logs / commit messages / bus / READMEs / public surfaces)
- **No Co-Authored-By** in commits (breaks Vercel billing)
- **Verify push before claiming on origin/main** (`git log origin/main..HEAD --oneline` should be empty before any "shipped" claim)
- **Two judges by default** (Sonnet + GPT-4o) on cross-family fine-tune evaluations
- **Three-gate framework** for production promotion: capability + voice + safety, all three must pass
- **Semantic-grade by default** for cross-family substring magnitudes
- **NPU Mid is INT8-only**; FP recipes pin to High
- **Skippy training content stays in Skippy deck**; other decks reference rather than reproduce

---

# Surface 1 — personal-ai-framework (Skippy) at v5.10.0

**Bus tag:** `[docs]` · **Working dir:** `~/Documents/GitHub/personal-ai-framework`

## 1. README refresh

Commit `159ef88` on personal-ai-framework origin/main. Pre-v5.10.0 README badged v5.9.2 with 14B-era perf numbers (85-140 tok/s sustained, 9.2 GB VRAM) — stale since the v4 campaign moved production to 7B v4. Rewrote the **Performance** section to lead with current 7B v4 measurements (180-215 tok/s sustained, 183.9 median, ~5.5 GB VRAM at 16K ctx, 30-120 ms TTFT, 1.0-2.0 s e2e on 200-token answer) + an explanatory paragraph naming the 14B v4 candidate's fabrication failure (76.2% headline but 0/3 on `made_up_peripheral`) with pointers to `docs/skippy-white-paper.md` + `docs/recipe-taxonomy.md`. Version history table appended with v5.10.0 entry (one paragraph summarizing methodology + research arc since v5.9.2; explicitly notes "No application-feature changes — engine is unchanged").

## 2. v5.10.0 annotated tag + GitHub release

Tag points at `159ef88`. Tag message comprehensively captures recovery state across methodology arc (gotcha-#7 closure, two-factor predictor N=7, cross-family v4 campaign), new tooling (`eval/regrade_semantic.py` + `eval/voice_metrics.py` + `eval/compare_accuracy_runs.py` + `eval/build_sizer_bundle.py` + `eval/quantize_w8a8.py`), new documentation (white paper, recipe taxonomy, private-anchor spec, presenter script, GOTCHA_7_RESOLUTION, Claude Bus protocol, RunPod runbooks, reviewer follow-up docs), deck (18 → 29 slides), production state, validated recipe cells, engine-unchanged statement.

GitHub release: https://github.com/kylefoxaustin/personal-ai-framework/releases/tag/v5.10.0

Release notes mirror the tag message in markdown — methodology arc bullets, new tooling, new docs, 29-slide deck structure, production decision, validated recipe cells (dense 7B-14B, MoE attention+router), performance summary, engine-unchanged statement.

## 3. Recovery memory

`project_v510_checkpoint.md` in `[docs]` memory store. Captures: what release captures (no-feature-work recovery snapshot), full methodology arc table with commit hashes (`f1de271` Gemma → `e0488e3` Yi → `0d07c51` Phi-4 → `3947c0a` reviewer closure → `ef46d99` semantic regrader → `706b0a4` bulk regrade → `5db0a0c` Qwen-bias closure), new tooling file map, new documentation file map, 29-slide deck structure with load-bearing slides marked, production decision (7B v4 ships / 14B v4 documented / MoE attention+router), cross-app coordination matrix, private-anchor-secrets spec pointer, reciprocal cross-references (Keyhole Slide 48 ↔ our Slide 18), NPU silicon canonical specs (deck Slide 11 is golden), why-v5.10.0-vs-v5.9.3 explanation, open items, recovery instructions, "do not update this memory" guard. Indexed in MEMORY.md.

## 4. What's at v5.10.0

### Methodology arc (load-bearing content)

| Finding | Where it lives |
|---|---|
| Gotcha-#7 closure: N=7 two-factor predictor reviewer-closed | `docs/GOTCHA_7_RESOLUTION.md` |
| Cross-family v4: Gemma +3.2 / Mistral −3.8 / Llama −3.2 / Yi −28.6 / Phi-4 −1.6 | `eval/results/acc_diff_*.md` + `docs/skippy-white-paper.md` |
| Qwen-family substring bias (campaign's most valuable methodology output) | `eval/regrade_semantic.py` + `eval/results/semantic_regrade_catalog.md` |

**Two-factor predictor (reviewer-final at N=7):** Substring lift requires ceiling reasoning (6/6) OR family-match to corpus source. Coverage: 3 of 4 corners measured; Qwen × floor-reasoning corner predicted-lift but untested.

**Substring-reliability framework:** base-vs-base at temp=0 reliable; base-vs-FT at temp=0 direction-only; base-vs-FT at temp>0 NO; cross-family intermediate-reasoning FT magnitude unreliable.

### Deck structure (29 slides; load-bearing in **bold**)

1. Title — Qwen 2.5 7B v4 production
2. Executive summary
3. System block diagram
4–6. Data flow: inference / agent / memory+RLHF
7. Measured KPIs (now leads with 7B v4 production)
8. BW math (134.4 GB/s peak / 100.8 GB/s usable @ 75% util)
9. Model catalog
10. Target NPU case study
11. **Compute tiers (Mid+High share 8.4 GT/s bus; decode BW-bound, TTFT compute-bound) — canonical NPU-tier framing**
12–13. MoE memory model / 3-tier table
14. Vendor claim reconciliation
15. Workload fit
16. Platform sizing
17. **Dense vs MoE BW physics validation (2 independent stacks within ~10%)**
18. **Skippy ↔ Keyhole cross-reference (NEW — reciprocal with Keyhole Slide 48)**
19. v4 campaign final (12 rows)
20. Cross-family baselines (N=7 stock bases)
21. Two-factor methodology + substring-reliability
22. **Process arc — six framings in <60 hrs**
23–25. Fabrication problem / options / Skippy choice
26. Recipe taxonomy framework
27. Voice gate
28. **Headline erosion — six methodology improvements retired the v4 capability claim**
29. Key takeaways

### Key file map

- Deck builder: `scripts/build_use_case_deck.py` (29-slide source)
- Deck rebuild pipeline: `scripts/rebuild_deck.sh` (build → corporate-template brand → Drive push)
- White paper: `docs/skippy-white-paper.md` (Findings 1-4 + per-regime substring reliability matrix)
- Recipe taxonomy: `docs/recipe-taxonomy.md` (6-dim recipe tuple + customer template)
- Private-anchor spec: `docs/private_anchor_secrets_spec.md` (canonical for all 4 surfaces)
- Presenter script: `docs/personal-ai-use-cases-presenter-script.md` (3 reviewer rounds; final-draft state)
- Gotcha #7 record: `docs/GOTCHA_7_RESOLUTION.md` (six framings audit trail)
- Bus protocol: `docs/CLAUDE_BUS.md`
- Semantic regrader: `eval/regrade_semantic.py` (GPT-4o; ~$0.66/eval)
- Voice metrics: `eval/voice_metrics.py` (post-process on existing eval JSONs)
- Sizer bundle: `eval/build_sizer_bundle.py` (bundles eval results)

### Discipline rules in force

- Zero measurement values in source tree / log / commit message / bus traffic
- KEY-not-VALUE for anchor-secrets data (treat like credentials)
- No Co-Authored-By (breaks Vercel billing)
- Two judges by default (Sonnet + GPT-4o) on cross-family fine-tune evaluations
- Three-gate framework: capability + voice + safety, all three must pass for production promotion
- Semantic-grade by default for cross-family substring magnitudes
- Verify-shipped before relaying claims (per `feedback_verify_shipped_claims` memory)

## 5. Recovery from this tag

```bash
cd /home/kyle/Documents/GitHub/personal-ai-framework
git fetch --tags
git checkout v5.10.0
# OR to reset main:
git reset --hard v5.10.0
```

Drive artifacts at `gdrive:skippy_files/personal-ai-assistant/` (deck + presenter script) are content-identical to `docs/` at this tag — restored together = full state.

## 6. Open items NOT in this release

- **BW two-factor refinement** (share × efficiency on slides 8/10) — Kyle hasn't approved; left at single-factor 75% util in deck. Doesn't block.
- **NPU High relabel to "400 TOPS FP-capable"** — Kyle gating on INT8-Mid crunch settling; deck uses "NPU High" only. Doesn't block.
- **Llama 2 7B vendor claim 60 tok/s** — Kyle-owed TBD; deck treats as methodology-teaching example (how to gut-check any vendor TPS claim against memory physics). Doesn't block.
- **BW-share selector UI (100/75/50/25%)** below tier dropdown in sizer apps — post-Path-C roadmap. Doesn't block.

---

# Surface 2 — personal-ai-assistant-sizer (PAI sizer) at v1.0.0

**Bus tag:** `[pai-sizer]` · **Working dir:** `~/Documents/GitHub/personal-ai-assistant-sizer`

## 1. What's tagged and where

- **Tag:** v1.0.0 (annotated)
- **Commit:** `7e34df0`
- **Repo:** github.com/kylefoxaustin/personal-ai-assistant-sizer
- **Release:** https://github.com/kylefoxaustin/personal-ai-assistant-sizer/releases/tag/v1.0.0
- **Live deployment:** https://personal-ai-assistant-sizer.streamlit.app/ (password-gated; auto-redeploys on push to main; v1.0.0 == what's currently live)

## 2. Deployment

- **Streamlit Community Cloud** — auto-redeploys on push to main
- **Shared-password gate** via `st.secrets PASSWORD`
- **Reload trap:** `app.py` auto-reloads; `sizer/*.py` changes require manual reboot from Streamlit Cloud console
- **No GPU dependency** — pure projection math over vendored `sizer/sizer_bundle.json` (5090 measurements)

### Local-only state (NOT in the tag)

- `.streamlit/secrets.toml` — anchor-secrets file (real NPU silicon measurements). Gitignored. Recoverable by `cp` from sister-app keyhole-sizer's same file.
- Kyle pre-staged this file once; both sister apps share content.

## 3. What's in the app at v1.0.0

### Architecture surfaces

- **Three-layer perf composition:** anchor overlay > Phase 1 measured override > Phase 2 projection (`project_llm` two-floor MAX `bw_floor` vs `compute_floor`)
- **NPU tier ladder:** 6 presets — Low-LP4 / Low-LP5-32bit / Low-LP5-64bit / Low-LP5X / Mid / **High default** / 5090. Default index lands on NPU High (FP-capable) so first-render is green for the default model.
- **Memory-upgrade overlays** on Mid + High: LPDDR5T-11.2 / LPDDR6-12 / LPDDR6-14
- **LLM catalog:** 20 entries — 1 PROD (skippy_7b_v4) + 7 FT + 6 BASE + 6 PERF reference (3 quant-variant Q5/Q8 + 3 alternate-compute-path INT8/FP). Default: skippy_7b_v4.
- **CNN coverage:** none — PAI sizer is LLM-only; CNN anchors visible in standalone display only (no projection path).
- **Anchor coverage:** 9/9 LLM spec cells reachable from headline tile (via 3 perf-reference compute-path variants added in `ee41def`). CNN scoped out on PAI side per domain split.
- **7-tab UX** (anchored convergence with keyhole-sizer's 8-tab): Overview · Accuracy · Precision · Performance · KPIs · Cost · Data
- **4-state source taxonomy** (🟢/🟡/🟠/🔴) + measured_silicon_anchor 🟢 banner state added Phase 2
- **Model-role icons** in selectbox: 🚀 PROD · 🔬 FT · 📚 BASE · ⚙️ PERF · 🔴 incompatible-prefix. Sort: (compatible, role priority, original order).
- **Workload categories:** 5 profiles (short_chat, rag_qa, long_decode, meeting_summarization, agentic_roundtrip). PAI does NOT have a workload_multiplier — `decode_tok_s` is workload-invariant by design (BW-bound sustained rate); workload affects token counts → TTFT + decode duration. (Differs from keyhole-sizer which has 1.000× → 0.038× multipliers.)
- **methodology_version** = `'2026-05-11-semantic-regrade-shipped'` surfaced in Accuracy tab footnote + About expander

### Key file map

- Catalog: `sizer/npu_model.py` MODELS dict
- Hardware tier defs + projection math: `sizer/npu_model.py`
- Anchor loader: `sizer/npu_anchors.py`
- Precision taxonomy + retargeting cost + annualized lifecycle: `sizer/precision.py`
- Bundle loader + bundle_meta accessor: `sizer/measured.py`
- Main app: `app.py` (~2240 lines, 7-tab structure)
- Anchor example: `.streamlit/secrets.toml.example`
- 5090 measurements: `sizer/sizer_bundle.json` (regenerated by personal-ai-framework `eval/build_sizer_bundle.py`)

### Discipline rules in force

- KEY-not-VALUE for anchor-secrets data (treat like credentials)
- No measurement values in source / commits / bus / chat
- Commits authored by Kyle Fox (no Co-Authored-By trailers)
- Per-push authorization for direct pushes to main
- Verify push landed before claiming "shipped" (`git log origin/main..HEAD --oneline` empty = pushed)
- Streamlit reboot rule: `sizer/*.py` touches require manual reboot
- Don't auto-mirror keyhole-sizer UX restructures without confirmation — relaxed post `8818916` (unified tab paradigm)

### Cross-app coordination at checkpoint

- **NPU canonical spec:** PAI deck Slide 11 (golden across all 4 surfaces; LPDDR5X 8.4 GT/s × 128-bit on Mid + High; tier differentiates on COMPUTE only, not BW)
- **Anchor-secrets canonical spec:** `personal-ai-framework/docs/private_anchor_secrets_spec.md`
- **Tab structure** matches keyhole-sizer's 8-tab variant; we share 7 tabs (Overview · Accuracy · Precision · Performance · KPIs · Cost · Data); keyhole adds Stream scaling + Duty-cycle for vision domain, replaces Cost + Data with KPIs + Detail
- **Role icons + 4-state source taxonomy:** same scheme both apps
- **Discipline rules:** identical across both sizers + framework

## 4. Recovery from this tag

```bash
git fetch --tags
git checkout v1.0.0

# Optional: populate real silicon anchors locally
cp ~/Documents/GitHub/keyhole-sizer/.streamlit/secrets.toml .streamlit/secrets.toml
# (or leave secrets.toml absent; anchors fall back to projection on missing values)

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Cloud deploy: paste contents of `secrets.toml` into share.streamlit.io → Settings → Secrets.

## 5. Open items NOT in this release

- 2 unmigrated entries (qwen3-30b-a3b-q4-moe, qwen2.5-14b-q4-dense) still on substring grading — could be semantic-regraded if `[docs]` produces `_semantic.json`. Family pattern delta ~−3.2pp.
- `resnet50_w8` cell not in anchor-secrets spec — one-line schema add if Kyle wants 8-bit-weight ResNet data surfaced.
- 5 of 9 LLM spec cells reachable today; the other 4 (mid_int8 + high_int8 dense models) would need W8A8 dense MODELS variants if/when measurements arrive.
- Engine-extraction (v1.x roadmap) — refactor shared LLM/anchor/bundle layers into common engine package consumed by PAI sizer + keyhole-sizer + drone-repo sizer + future 4th sizer.

---

# Surface 3 — keyhole at v1.0.0

**Bus tag:** `[backend]` · **Working dir:** `~/Documents/GitHub/keyhole`

## 1. What this project is

**keyhole** is the edge AI video analytics platform: vision pipeline (FFmpeg → YOLO-seg → CLIP open-vocab labels → SQLite event store → optional LLM-NLQ over events) + measurement harness + 63-slide engineering deck. The repo is where bake-offs land, where the conceptual frame is held, and where the customer-facing deck variants are built. Not a hosted webapp — a measurement + documentation repo that also runs locally via a FastAPI server when consumed by keyhole-UI.

- **Repo:** https://github.com/kylefoxaustin/keyhole
- **Live:** none — local FastAPI server on `http://localhost:8777` when run
- **Working dir:** `~/Documents/GitHub/keyhole`
- **Bus tag:** `[backend]`
- **Companion apps:** `[sizer]` keyhole-sizer (Streamlit what-if sizer); `[docs]` personal-ai-framework (Skippy product / LLM training story); `[pai-sizer]` personal-ai-assistant-sizer (LLM-only sister sizer)

## 2. Tagged version on GitHub

- **Tag:** `v1.0.0`
- **Commit:** `a23af3b`
- **Tag date:** 2026-05-18
- **Release URL:** https://github.com/kylefoxaustin/keyhole/releases/tag/v1.0.0
- **Prior tag:** none — v1.0.0 is the inaugural tag

## 3. Major key points / dependencies

### Runtime versions

| What | Version |
|---|---|
| Python | 3.10+ (3.12+ recommended for SAM 3 + TRT 10.16) |
| PyTorch | 2.7+ with CUDA 12.x |
| TensorRT | 10.16 (Blackwell-capable; FP8 + INT8 compile paths) |
| ultralytics | for YOLO 11 + YOLO-seg variants |
| open-clip-torch | CLIP visual + text encoders |
| SAM 3 | from source (third_party/sam3, HuggingFace access required) |
| python-pptx | deck generation |
| FastAPI + uvicorn | HTTP API server |
| llama.cpp + GGUF | LLM inference path (Qwen3-30B-A3B Q4_K_M) |
| FFmpeg | frame extraction |
| Nsight Compute | DRAM measurement pipeline |

`requirements.txt` + manual `third_party/` installs. No Pipfile/poetry/pyproject.toml.

### Hardware envelope

- Reference platform: NVIDIA RTX 5090 (Blackwell SM 12.0, 32 GB GDDR7 @ 1792 GB/s), Intel i9-14900KF host
- Target silicon: 5 NPU tiers (Low-LP5-32bit / Low-LP5-64bit / Low-LP5X / Mid / High) + i.MX 95 ground truth
- Sister sizer twin: keyhole-sizer pulls from the same NPU tier model

### Deployment

- Not a hosted webapp. Decks + measurement artifacts ship via:
  - `my-stuff` repo (kylefoxaustin/my-stuff, branch `claude/conductor-dashboard-WnjwO`) — pptx artifacts
  - `gdrive:skippy_files/keyhole/look_here/` — curated reviewer folder
- FastAPI server runs locally at `http://localhost:8777` for keyhole-UI consumption

### Local-only state (NOT in the tag)

- `.streamlit/secrets.toml` — anchor-secrets file (measured NPU silicon performance), gitignored. Recover by `cp` from PAI sister-sizer's same file. Loader: `src/anchors/private_anchors.py` (never prints values; KEY-not-VALUE discipline).
- `data/output/` — entire dir is gitignored. Contains `keyhole_results.pptx`, `keyhole_deck_branded.pptx`, bake-off JSONs, ncu measurements. Recover decks from `my-stuff` repo or `gdrive:skippy_files/keyhole/look_here/`.
- `data/videos/` — input test footage, gitignored.
- `third_party/sam3/` — SAM 3 source install, gitignored (HuggingFace access required to clone).
- `data/output/keyhole_results_PRIVATE.pptx` — `--include-private` build output containing measured silicon values. Gitignored. NXP-internal destinations only — never my-stuff, never gdrive.

## 4. What's in the repo at v1.0.0

### Architecture surfaces

- **63-slide deck** with section structure: Opening (1-8) → Per-clip baselines (9-29) → Bandwidth physics (30-32) → Quantization journey (33-36) → Architectural pivot (37-39) → FP8 + TensorRT (40-47) → LLM identity + bake-off + duty cycle (48-50) → Cross-cutting LLM findings (51-52) → Community SAM 3 + ViT alternatives (53-57) → TRT takeaways (58) → ncu validation (59-61) → Roadmap + Summary (62-63). Optional private slide 64.
- **Two deck variants from same source:**
  - Plain (dark-bg) — `data/output/keyhole_results.pptx`, 63 slides
  - NXP-branded — `data/output/keyhole_deck_branded.pptx`, 63 slides via `pptx_template_converter` Strategy A theme swap
- **Bake-off catalog** — TRT YOLO INT8/FP8 (full-model on Blackwell, recall 1.000 @ IoU 0.998 vs FP16), TRT CLIP FP16/FP8 (3× speedup), mask-model bake-off (MobileSAM / EfficientSAM-tiny/small / YOLO-seg), EfficientSAM3 community variants (Apr 2026), YOLOE-26 one-model open-vocab, ViT alternatives what-if (RT-DETR-L / DETR-ResNet50 / OWLv2 / Grounding DINO), LLM 5090 anchor catalog (Qwen 7B / 32B / 30B-A3B MoE).
- **Measurement pipelines** — Nsight Compute DRAM/forward pipeline; end-to-end latency budget; i.MX 95 anchor (yolov8n-seg INT8 1080p = 32 ms / 29.2 FPS measured).
- **HTTP API** — FastAPI server at `http://localhost:8777`; endpoints in `API.md`. Consumed by keyhole-UI sister repo.
- **Anchor secrets system** — `--include-private` flag on `build_deck.py` surfaces measured silicon values from gitignored `.streamlit/secrets.toml` onto extra slide 64; loader at `src/anchors/private_anchors.py` never prints values.
- **Headline numbers:** SAM 3 baseline 0.4 FPS edge → Hybrid V2 TRT FP8 = 36 FPS at 720p on NPU Mid. 515× DRAM reduction (Nsight Compute measurement: 118,975 MB / 231 MB amortized).

### Key file map

- Deck generator: `scripts/build_deck.py` (supports `KEYHOLE_DECK_MERGE_TARGET=1` for branded conversion + `--include-private` for anchors slide)
- Bake-off harnesses: `scripts/bakeoff_*.py` (trt_yolo, trt_clip, llm_anchors, concurrency, smoothquant, fp8_yolo, yolo_conv_quant, yoloe26, efficientsam3*)
- ncu profile pipeline: `scripts/profile_ncu.py`
- Branded variant patcher (obsolete, kept for reference): `scripts/update_branded_deck.py`
- Anchor loader: `src/anchors/private_anchors.py`; schema: `.streamlit/secrets.toml.example`
- HTTP API: `src/api/server.py`; CLI entry: `src/main.py`
- Vision pipelines: `src/detect/yolo.py`, `src/detect/sam3_detect.py`, `src/detect/hybrid_v2.py`
- Edge projection: `src/emulate/npu_emulator.py`, `src/emulate/sam3_reference.py`
- Docs: `docs/PRESENTER_SCRIPT.md` (45-60 min walkthrough), `docs/CHANGES_2026-05-17.md` (reviewer orientation), `docs/ALIGNMENT_PLAN.md` (alignment history — ALL PHASES DONE), `docs/PRIVATE_DECK.md` (operator discipline)

### Discipline rules in force

- **KEY-not-VALUE** for anchor-secrets data (treat like credentials; no measurement values in source / logs / commit messages / bus / READMEs / public surfaces)
- **No Co-Authored-By** in commits (breaks Vercel billing)
- **Verify push before claiming on origin/main** (`git log origin/main..HEAD` before any bus broadcast claiming a commit is shipped)
- **`keyhole_results_PRIVATE.pptx` discipline:** NXP-internal destinations only; never my-stuff, never gdrive
- **Skippy training content stays in Skippy deck** (per conceptual frame); Keyhole references rather than reproduces
- **NPU Mid is INT8-only** per PAI deck slide 11 golden; FP recipes pin to High (not a same-bus-but-Mid-FP-capable framing — that was a same-day drift Kyle corrected)

### Cross-app coordination at checkpoint

- **NPU canonical spec:** PAI deck Slide 11 (golden across all 4 surfaces; Mid INT8-only, Mid + High share 128-bit LPDDR5X @ 8.4 GT/s, High differentiates on compute + capacity + TDP)
- **Anchor-secrets canonical spec:** `personal-ai-framework/docs/private_anchor_secrets_spec.md`
- **Reciprocal cross-reference:** PAI deck slide 18 ↔ Keyhole deck slide 48 (mirror pattern: "what this deck is / where the other deck lives")
- **Conceptual frame:** Keyhole = vision platform; LLM = optional surface (Skippy artifact unmodified); three operational modes (vision-only / vision+LLM / LLM-only)

## 5. Recovery from this tag

```bash
git fetch --tags
git checkout v1.0.0
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SAM 3 from source (HuggingFace access required)
mkdir -p third_party && cd third_party
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e ".[notebooks]"
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
cd ../..
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install open-clip-torch
wget -O weights/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
hf auth login

# Optional — recover anchor secrets from PAI sister-sizer
cp ~/Documents/GitHub/personal-ai-assistant-sizer/.streamlit/secrets.toml .streamlit/secrets.toml

# Rebuild plain deck (63 slides)
python scripts/build_deck.py
# Output: data/output/keyhole_results.pptx

# Rebuild branded deck (NXP corporate template)
KEYHOLE_DECK_MERGE_TARGET=1 python scripts/build_deck.py
cp data/output/keyhole_results.pptx \
  ~/Documents/GitHub/pptx_template_converter/input/keyhole_merge_ready.pptx
cd ~/Documents/GitHub/pptx_template_converter
python convert.py \
  --input    input/keyhole_merge_ready.pptx \
  --template template/corporate_template.pptx \
  --output   output/keyhole_deck_branded.pptx \
  --color-map mappings/keyhole_to_corporate.json
cp output/keyhole_deck_branded.pptx \
  ~/Documents/GitHub/keyhole/data/output/keyhole_deck_branded.pptx
cd ~/Documents/GitHub/keyhole
python scripts/build_deck.py    # restore plain variant

# Build private variant (optional, NXP-internal only)
python scripts/build_deck.py --include-private
# Output: data/output/keyhole_results_PRIVATE.pptx (64 slides incl. anchors slide)

# Start HTTP API server (consumed by keyhole-UI or any HTTP client)
python -m src.main serve
```

For "I lost the deck artifacts entirely," pull from `gdrive:skippy_files/keyhole/look_here/` or the `my-stuff` repo.

## 6. Open items NOT in this release

- **INT8 CLIP port** (~1-2 weeks of focused work per roadmap slide 60) — would unlock the full Hybrid V2 pipeline on NPU Mid silicon. Currently FP recipes pin to High.
- **Real Mid-class NPU silicon vision anchor** (KH-P2-001) — i.MX 95 is the only edge measurement we have; everything else is 5090-projected via BW ratio.
- **Branded deck slide-1 title-date parameterization** — currently build-time date; would parameterize for fixed customer-meeting dates if needed.
- **Engine extraction itself** — the cross-repo work this tag exists to enable. Drone use case + shared "engine" pattern across personal-ai-framework / keyhole / drone-repo / future 4th use case.

---

# Surface 4 — keyhole-sizer at v1.0.0

**Bus tag:** `[sizer]` · **Working dir:** `~/Documents/GitHub/keyhole-sizer`

## 1. What this project is

**keyhole-sizer** is a Streamlit web app — an interactive NPU sizing sandbox for the Keyhole edge-AI bake-off findings. Pick an NPU tier (or build a custom one), a vision pipeline, concurrent stream count, and optional LLM co-existence; watch live FPS / tok/s / VRAM-fit / duty-cycle projections.

- **Repo:** https://github.com/kylefoxaustin/keyhole-sizer
- **Live:** https://keyhole-sizer.streamlit.app (Streamlit Community Cloud)
- **Working dir:** `~/Documents/GitHub/keyhole-sizer`
- **Bus tag:** `[sizer]`
- **Companion app:** `[pai-sizer]` (personal-ai-assistant-sizer) — LLM-only sister

## 2. Tagged version on GitHub

- **Tag:** `v1.0.0`
- **Commit:** `21debd2`
- **Tag date:** 2026-05-18
- **Release URL:** https://github.com/kylefoxaustin/keyhole-sizer/releases/tag/v1.0.0
- **Prior tag:** none — v1.0.0 is the inaugural tag

## 3. Major key points / dependencies

### Runtime versions

| What | Version |
|---|---|
| Python | 3.10.12 (Streamlit Cloud uses 3.10/3.11) |
| Streamlit | >=1.40.0 (installed locally: 1.56.0) |
| pandas | >=2.2.0 |
| numpy | >=1.26.0 |
| plotly | >=5.24.0 |
| openpyxl | >=3.1.0 |

`requirements.txt` is the canonical pin. No Pipfile/poetry/pyproject.toml.

### Deployment

- **Streamlit Community Cloud** — auto-redeploys on push to main
- **Shared-password gate** via `st.secrets PASSWORD` (bypassed locally when secret absent)
- **Reload trap:** `app.py` auto-reloads; `sizer/*.py` changes require manual reboot from Streamlit Cloud console
- **No GPU dependency** — pure projection math on vendored JSON (`sizer/sizer_bundle.json`)

### Local-only state (NOT in the tag)

- `.streamlit/secrets.toml` — anchor-secrets file (real NPU silicon measurements), gitignored. Recover by `cp` from PAI sister-sizer's same file.

## 4. What's in the app at v1.0.0

### Architecture surfaces

- **Three-layer projection composition:** anchor overlay > Phase 1 measured override > Phase 2 projection (`max(bw_floor, compute_floor)` + overhead)
- **NPU tier ladder:** 6 presets + Custom (i.MX 95 / Low-LP5-64bit / Low-LP5X / Mid / **High default** / 5090 / Custom). Default index=4 (NPU High, FP-capable) so default LLM is compatible on first load.
- **Memory-upgrade overlays** on Mid + High: LPDDR5T-11.2 / LPDDR6-12 / LPDDR6-14
- **LLM catalog:** 17 entries (1 PROD + 7 FT + 6 BASE + 3 PERF reference). Default: skippy_7b_v4
- **Vision pipelines:** 23 entries (21 originals + 2 4-bit-weight CNN variants)
- **Anchor coverage:** 9/9 LLM + 6/6 CNN spec cells reachable from headline tiles
- **8-tab UX** (mirrors PAI sizer): Overview · Accuracy · Precision · Performance · Stream scaling · Duty-cycle · KPIs · Detail
- **4-state source taxonomy** (green/yellow/orange/red) + measured_silicon_anchor
- **Model-role icons** in selectbox: PROD · FT · BASE · PERF · incompatible-prefix
- **Workload-pattern multipliers:** 5 categories (plain_chat 1.000× → cold_start 0.038×, 26× spread)
- **METHODOLOGY_VERSION** = `'2026-05-11-semantic-regrade-shipped'`

### Key file map

- Catalog: `sizer/llm_models.py`
- Pipelines + Hardware tier defs + projection math: `sizer/npu_model.py`
- Anchor loader: `sizer/npu_anchors.py`
- Precision taxonomy: `sizer/precision.py`
- KPI breakdown: `sizer/kpi_breakdown.py`
- Platform-budget exports: `sizer/platform_budget.py` + `scripts/export_platform_*.py`
- Main app: `app.py` (~2400 lines)
- Anchor example: `.streamlit/secrets.toml.example`

### Discipline rules in force

- KEY-not-VALUE for anchor-secrets data (treat like credentials)
- Anchor-overlay must compose with `workload_multiplier()`
- Verify push before claiming "shipped to origin/main"
- Three-site pipeline registration (5-site for anchor-reachable CNNs) enforced by startup assertion
- No Co-Authored-By (breaks Vercel billing)
- "DEFAULT" not "SHIPPING" terminology

### Cross-app coordination at checkpoint

- **NPU canonical spec:** PAI deck Slide 11 (golden across all 4 surfaces)
- **Anchor-secrets canonical spec:** `personal-ai-framework/docs/private_anchor_secrets_spec.md`
- **Tab structure** verbatim-mirrors PAI sizer; same source taxonomy, role icons, workload categories

## 5. Recovery from this tag

```bash
git fetch --tags
git checkout v1.0.0
cp ~/Documents/GitHub/personal-ai-assistant-sizer/.streamlit/secrets.toml .streamlit/secrets.toml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Live deployment auto-redeploys on push to main; v1.0.0 is already what's live.

## 6. Open items NOT in this release

- **KH-P2-001** — real Mid-class NPU silicon vision anchor (hardware-access dependent)
- **KH-P3-003** — 5090-side W4 bake-offs for methodology validation (parked; silicon-anchor takes precedence)
- **resnet50_w8 spec gap** — one-line schema addition if needed

---

# Engine extraction readiness

All four surfaces tagged + released + memory-checkpointed. Kyle has 100% recoverable state across the ecosystem. Ready for engine-extraction kickoff.

**The drone use case** (third application of the engine pattern) will incorporate:
- Skippy's LLM artifact + fine-tuning recipe (from personal-ai-framework v5.10.0)
- Keyhole's vision pipeline (from keyhole v1.0.0)
- Drone-specific additions

**The engine pattern** is the common base that all four future use cases (personal AI assistant, Keyhole, drones, future 4th) will build on. Engine extraction is cross-repo refactoring; this checkpoint exists so any divergence during that work is recoverable.

**Recovery target for the whole ecosystem:** the four tags above. Restore each repo to its tag, restore each `.streamlit/secrets.toml` from the cross-app `cp` chain, restore Drive artifacts from `gdrive:skippy_files/personal-ai-assistant/` and `gdrive:skippy_files/keyhole/look_here/`. Anchor-secrets values themselves never enter source / git / chat / bus per the standing discipline rule — they live only in local `.streamlit/secrets.toml` files and in Streamlit Cloud Secrets (encrypted at rest).
