# Phase 5 scope recon — Skippy framework retrofit (no-op, upstream authority)

**Date:** 2026-05-23
**From:** ratchet consolidation, phase 5 (Skippy framework) read-and-acknowledge
**Decision:** Option **α′** approved — no-op retrofit, no pin, tag v5.10.1.

---

## 1. Where this fits

`ratchet` is a pure-Python edge-SoC sizing engine being consolidated from four
production surfaces. Progress when phase 5 began:

- **Phase 1 — DONE.** ratchet v0.2.0 → v0.2.4 (Amendments 3/4/5/6 during the
  retrofits).
- **Phase 2 — DONE.** PAI sizer v1.1.0 (Option C: adopt ratchet at the
  Hardware/TIERS/loader/capability layer).
- **Phase 3 — DONE.** keyhole-sizer v1.1.0 (Option C analog, D5(i) full Hardware
  adoption + surface-side adapters).
- **Phase 4 — DONE.** keyhole backend v1.0.1 (Option α: no-op + future-use pin
  + documented divergence).
- **Phase 5 — THIS.** Skippy framework (this repo, `personal-ai-framework`).

The design (§1, §12) predicted phase 5 as *"possibly the smallest retrofit
despite the framework's overall complexity. Skippy uses ratchet only for
cross-surface canonical specs and sizer-bundle integration. Most Skippy code is
unrelated to ratchet's domain."*

---

## 2. Empirical recon

### 2a. Skippy doesn't import ratchet

Confirmed with anchored-import greps across `pipeline/ eval/ scripts/
training/ web/`:

```
$ grep -rn "^from ratchet\|^import ratchet\|^from sizer\|^import sizer\|^from keyhole\|^import keyhole" \
    pipeline eval scripts training web
(no output)
```

**Zero actual Python imports** of ratchet, sizer, or keyhole anywhere in Skippy
code. The five "near-miss" substring hits were either documentation comments
referencing "personal-ai-assistant-sizer" or generated JSON artifacts — not
imports.

### 2b. No engine-equivalent code exists in Skippy

Searching for ratchet-target symbols (`class Hardware` / `class HardwareSpec` /
`^TIERS\s*=` / `npu_anchors` / `measured_llm_q4_decode_tok_s` /
`peak_tops_int8` / `capability_levels` / `tensor_native` / `sm120`) returned
**no code matches** in `pipeline/ eval/ scripts/ knowledge/ training/ docs/
web/ tests/`. The hits were:

- `docs/private_anchor_secrets_spec.md` — the canonical *spec* (Skippy
  authoring), not engine code.
- `docs/ECOSYSTEM_CHECKPOINT_2026-05-18.md` — coordination doc.
- `eval/results/sizer_bundle.json` + `sizer_bundle_prelim.json` — data
  artifacts produced for downstream consumption.
- `knowledge/emails/*` — random text matching substrings in archived emails;
  noise.

**Skippy has no `Hardware` / `HardwareSpec` class, no `TIERS` ladder, no anchor
loader, no capability tables, no projection function.**

### 2c. What Skippy actually produces

Two cross-surface artifacts authored upstream of ratchet and the sizers:

1. **`docs/private_anchor_secrets_spec.md`** — the canonical anchor-secrets
   schema (`tokps`, `peak_bw_gbps`, `bw_share_frac`, `bw_efficiency_frac`,
   `source`, `measured_date` and the CNN parallel). ratchet's loader, PAI's
   loader, and keyhole-sizer's loader **all conform to this spec**. Skippy is
   the source of truth.
2. **`eval/build_sizer_bundle.py`** (197 lines) — produces
   `eval/results/sizer_bundle.json`, the bundle PAI sizer's `measured.py`
   consumes to populate `RTX_5090_REFERENCE.measured_llm`. Cross-surface
   integration that flows out of Skippy.

That's the entire surface area of "Skippy ↔ ratchet ecosystem" — both
**producer-side**.

---

## 3. The relationship is structurally upstream

The data flow is **Skippy → consumers** (ratchet, PAI sizer, keyhole-sizer),
not Skippy ← ratchet:

```
   Skippy (this repo)
      │
      ├─ authors anchor-secrets schema   ──►  ratchet (implements)
      │                                       │
      │                                       ├─►  PAI sizer (consumes)
      │                                       └─►  keyhole-sizer (consumes)
      │
      └─ produces sizer_bundle.json      ──►  PAI sizer (measured.py)
```

There is **no consumer interface to retrofit**. ratchet's anchor loader
conforms to Skippy's spec, not the other way around.

---

## 4. Decision — Option α′ approved (reviewer ruling 2026-05-23)

**No code adoption. No pin. Tag v5.10.1.**

- **Why no pin:** pinning a dependency Skippy doesn't import would
  misrepresent the upstream-producer relationship as a downstream-consumer
  one. A future contributor auditing `requirements.txt` would see
  `ratchet>=0.2.4,<0.3.0`, grep for `import ratchet`, find nothing, and either
  (a) think the dependency is dead and remove it, or (b) assume non-obvious
  dynamic loading. Both bad. CLAUDE.md is the truthful, durable hook for the
  upstream-producer relationship.
- **Why still tag:** the discipline pattern is *"each phase ends with a tag."*
  Skipping the tag would leave a gap in the migration history that future
  contributors would read as incomplete work. The four phases should show as
  four distinct checkpoints with their **actual** scopes — Hardware adoption,
  Hardware adoption with vision adapters, no-op with pin, no-op without pin —
  not as "Skippy wasn't done."
- **Discipline symmetry isn't a value in itself.** Each phase's checkpoint
  reflects its actual relationship to ratchet:
  - Phases 2/3 *pin because they import*.
  - Phase 4 *pins as a future-use hook* for a backend that could consume.
  - Phase 5 *doesn't pin because the relationship is structurally upstream*.
  Three shapes for three relationships — information, not inconsistency.

---

## 5. Ecosystem shape after phase 5

Four-surface mapping with full empirical clarity:

| Surface | Relationship to ratchet | Tag |
|---|---|---|
| **PAI sizer** | consumer (Hardware/TIERS/loader/capability) | v1.1.0 |
| **keyhole-sizer** | consumer (same + vision adapters) | v1.1.0 |
| **keyhole backend** | sibling repo, different domain — future-use pin | v1.0.1 |
| **Skippy framework** | upstream author of the anchor-secrets spec | v5.10.1 |

The ecosystem isn't a flat "four surfaces share an engine" — it's a
directional graph: Skippy at the top (authoring specs), ratchet in the middle
(implementing them and adding the tier-registry + projection layer on top),
the two sizers as consumers, and the backend as a parallel sibling. The
retrofits made these relationships explicit and machine-readable.

The original design's "four surfaces with duplicated engine code" framing
overstated the duplication; reality is two consumers + one sibling + one
upstream. That's the *correct* shape — calibration, not a problem.

---

## 6. What did NOT happen in this retrofit

- No edits to `eval/build_sizer_bundle.py` (the bundle producer continues to
  work as-is; PAI consumes its output).
- No edits to `docs/private_anchor_secrets_spec.md` (Skippy remains the
  authority; ratchet conforms).
- No `requirements.txt` changes (no ratchet dependency added).
- No engine code, capability tables, or anchor-loader code introduced.

This was the right shape for phase 5: acknowledge the upstream-producer
relationship, document it durably, and close the migration checkpoint.
