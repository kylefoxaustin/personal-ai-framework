#!/usr/bin/env python3
"""
Build a single XLSX bundle of all Skippy testing/perf/recipe data, sized for a
fresh Claude (or any external reviewer) to inspect raw numbers without needing
to clone the repo or read individual eval JSONs.

Output: docs/skippy-data-bundle.xlsx — 8 sheets:
  1. index               — what each sheet contains, how to read it
  2. models              — every base + fine-tune we evaluated (arch, size, GGUF, training cost)
  3. eval_headlines      — pass rate per (model, eval-run) — one row per JSON in eval/results/
  4. eval_per_category   — long format: (model × category) → pass / n / rate
  5. voice_metrics       — voice gate measurements (length, bullets, bolds, emojis, opener-boilerplate)
  6. perf_5090           — RTX 5090 measurements: decode tok/s, prefill tok/s, RAG total
  7. recipe_matrix       — the 8 filled cells in the recipe taxonomy with all 8 dimensions
  8. methodology         — eval prompt count, RAG config, hardware, sample counts, gotchas

Reproducibility: the eval_headlines + eval_per_category sheets are scraped
fresh from eval/results/acc_*.json on every run. The other sheets contain
human-curated data (voice metrics, perf measurements, recipe taxonomy) and
need to be updated in this file when those data sets change.

Usage:
    python3 scripts/build_data_bundle.py
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "skippy-data-bundle.xlsx"


# ============================================================
# Sheet 2 — models metadata (hardcoded, curated)
# ============================================================

MODELS_ROWS = [
    # name, family, kind, arch, size_b_total, size_b_active, gguf_q4km_gb,
    # training_hardware, training_cost_usd, training_walltime_min, role
    ("Qwen 2.5 7B Instruct (stock)",       "Qwen",    "base",      "dense", 7.6,  7.6, 4.68, "—",       0,    0,   "base reference"),
    ("Qwen 2.5 14B Instruct (stock)",      "Qwen",    "base",      "dense", 14.7, 14.7, 8.7,  "—",       0,    0,   "base reference (also prior production)"),
    ("Qwen 2.5 32B Instruct (stock)",      "Qwen",    "base",      "dense", 32.5, 32.5, 19.0, "—",       0,    0,   "base reference"),
    ("Qwen 3 30B-A3B Instruct-2507",       "Qwen",    "base",      "MoE",   30.0, 3.0,  18.0, "—",       0,    0,   "base reference (MoE, 8/128 experts)"),
    ("Mistral 7B v0.3 Instruct",           "Mistral", "base",      "dense", 7.25, 7.25, 4.37, "—",       0,    0,   "base reference (cross-family)"),
    ("Llama-3.1 8B Instruct",              "Llama",   "base",      "dense", 8.03, 8.03, 4.92, "—",       0,    0,   "base reference (cross-family)"),
    ("Skippy 7B v1",                        "Qwen",    "fine-tune", "dense", 7.6,  7.6,  4.68, "5090",    0,    85,  "iteration v1 — 75% headline, but rambles"),
    ("Skippy 7B v2",                        "Qwen",    "fine-tune", "dense", 7.6,  7.6,  4.68, "5090",    0,    90,  "iteration v2 — fixed pad bug, broke other"),
    ("Skippy 7B v3",                        "Qwen",    "fine-tune", "dense", 7.6,  7.6,  4.68, "5090",    0,    70,  "iteration v3 — architectural rewrite, over-refuses"),
    ("Skippy 7B v4 ★ PRODUCTION",          "Qwen",    "fine-tune", "dense", 7.6,  7.6,  4.68, "5090",    0,    46,  "current production"),
    ("Skippy 14B v1",                       "Qwen",    "fine-tune", "dense", 14.7, 14.7, 8.7,  "5090",    0,    240, "early 14B candidate"),
    ("Skippy 14B v4",                       "Qwen",    "fine-tune", "dense", 14.7, 14.7, 8.7,  "5090",    0,    180, "best headline; fabricates fictional peripherals"),
    ("Skippy 32B v1",                       "Qwen",    "fine-tune", "dense", 32.5, 32.5, 19.0, "H100",   25,    300, "32B pre-clean attempt"),
    ("Skippy 32B v4 (CLEAN, 2 ep)",         "Qwen",    "fine-tune", "dense", 32.5, 32.5, 19.0, "H100",   30,    420, "regresses −4.6pp from base; corpus-too-small"),
    ("Skippy MoE v4 (attention-only)",      "Qwen",    "fine-tune", "MoE",   30.0, 3.0,  18.0, "H100",   15,    300, "catastrophic regression on multihop"),
    ("Skippy MoE v4 + router",              "Qwen",    "fine-tune", "MoE",   30.0, 3.0,  18.0, "H100",   18,    300, "router LoRA recovers reasoning — recommended MoE recipe"),
    ("Skippy MoE v4 + router + experts",    "Qwen",    "fine-tune", "MoE",   30.0, 3.0,  18.0, "H100",   25,    360, "expert LoRA over-fits 6.5K-example corpus"),
    ("Skippy Mistral v4",                   "Mistral", "fine-tune", "dense", 7.25, 7.25, 4.37, "5090",    0,    48,  "cross-family — recipe regresses (NEW finding)"),
]

MODELS_COLS = ["model", "family", "kind", "architecture", "params_total_b",
               "params_active_b", "gguf_q4km_gb", "training_hw",
               "training_cost_usd", "training_walltime_min", "role"]


# ============================================================
# Sheets 3 + 4 — eval headlines + per-category (scraped from JSONs)
# ============================================================

# Map raw JSON `name` field → human-friendly model name used elsewhere in the bundle
NAME_MAP = {
    "baseline-qwen25-7b-base-v2-rag":                       "Qwen 2.5 7B Instruct (stock)",
    "baseline-qwen25-32b-instruct-v2-rag":                  "Qwen 2.5 32B Instruct (stock)",
    "baseline-qwen3-30b-a3b-instruct-2507-v2-rag":          "Qwen 3 30B-A3B Instruct-2507",
    "baseline-mistral-7b-instruct-v0.3-v2-rag":             "Mistral 7B v0.3 Instruct",
    "baseline-llama-3.1-8b-instruct-v2-rag":                "Llama-3.1 8B Instruct",
    "candidate-kyle-qwen25-7b-v1-v2-rag":                   "Skippy 7B v1",
    "candidate-kyle-qwen25-7b-v2-v2-rag":                   "Skippy 7B v2",
    "candidate-kyle-qwen25-7b-v3-v2-rag":                   "Skippy 7B v3",
    "candidate-kyle-qwen25-7b-v4-v2-rag":                   "Skippy 7B v4 ★ PRODUCTION",
    "candidate-kyle-qwen25-14b-v1-v2-rag":                  "Skippy 14B v1",
    "candidate-kyle-qwen25-32b-v1-v2-rag":                  "Skippy 32B v1",
    "candidate-kyle-qwen25-32b-v4-clean-v2-rag":            "Skippy 32B v4 (CLEAN, 2 ep)",
    "candidate-kyle-qwen3-30b-a3b-v4-v2-rag":               "Skippy MoE v4 (attention-only)",
    "candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag":        "Skippy MoE v4 + router",
    "candidate-kyle-qwen3-30b-a3b-full-v1-v2-rag":          "Skippy MoE v4 + router + experts",
    "candidate-kyle-mistral-7b-v4":                          "Skippy Mistral v4",
    # Pre-v4 historical (April 2026 quantization + Thinking-MoE bake-off era)
    "reference-dense-q4km-v2-rag":                            "[pre-v4] Qwen 2.5 14B Q4_K_M (reference)",
    "reference-dense-q4km":                                   "[pre-v4] Qwen 2.5 14B Q4_K_M (early reference)",
    "reference-qwen25-14b-vanilla":                           "[pre-v4] Qwen 2.5 14B vanilla",
    "reference-moe-q4km-v2-rag":                              "[pre-v4] Skippy MoE early FT (pre-v4 recipe)",
    "reference-moe-q4km":                                     "[pre-v4] Skippy MoE early FT (no RAG)",
    "reference-moe-q4km-v2":                                  "[pre-v4] Skippy MoE early FT (v2 prompts, no RAG)",
    "reference-moe-q4km-rerun":                               "[pre-v4] Skippy MoE early FT (rerun)",
    "reference-dense-fp16-v1-vllm":                           "[pre-v4] Skippy 14B v1 FP16 (vllm)",
    "reference-dense-fp16-v2-vllm":                           "[pre-v4] Skippy 14B v1 FP16 v2 prompts (vllm)",
    "candidate-dense-fp8-v1":                                 "[pre-v4] Skippy 14B v1 FP8 W8A8",
    "candidate-dense-fp8-v2":                                 "[pre-v4] Skippy 14B v1 FP8 W8A8 v2",
    "candidate-dense-fp8-v2-rag":                             "[pre-v4] Skippy 14B v1 FP8 W8A8 v2-rag",
    "candidate-dense-q8-v2-rag":                              "[pre-v4] Skippy 14B v1 Q8 v2-rag",
    "candidate-qwen25-14b-q8":                                "[pre-v4] Qwen 2.5 14B Q8 stock",
    "candidate-moe-thinking-v2-rag":                          "[pre-v4] Qwen3-30B-A3B-Thinking-2507 stock",
}


def scrape_eval_jsons():
    """Walk eval/results/acc_*.json and return (headlines_df, per_category_df)."""
    headline_rows, cat_rows = [], []
    for path in sorted(RESULTS_DIR.glob("acc_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Skip diff files (acc_diff_*.json) — they have a different schema
        if "diffs" in data and "prompts" not in data:
            continue
        raw_name = data.get("name", path.stem)
        model = NAME_MAP.get(raw_name)
        if not model:
            # Unmapped — include with raw name so reviewer sees it exists
            model = f"[unmapped] {raw_name}"
        summary = data.get("summary", {})
        passed = summary.get("passed")
        total = summary.get("total") or 132
        rate = summary.get("pass_rate")
        if passed is None:
            # Older JSON format — sum from prompts
            passed = sum(p.get("aggregate", {}).get("pass_n", 0) for p in data.get("prompts", []))
            rate = passed / total if total else 0.0
        headline_rows.append({
            "model": model,
            "raw_eval_name": raw_name,
            "timestamp": data.get("timestamp", ""),
            "samples_per_prompt": data.get("samples_per_prompt", 3),
            "use_rag": data.get("config", {}).get("use_rag", True),
            "passes": passed,
            "total": total,
            "pass_rate_pct": round(rate * 100, 1) if rate else None,
            "json_file": path.name,
        })
        # Per-category
        by_cat = defaultdict(lambda: [0, 0])
        for p in data.get("prompts", []):
            cat = p.get("category") or "unknown"
            agg = p.get("aggregate", {})
            n = agg.get("total", len(p.get("samples", [])))
            by_cat[cat][0] += agg.get("pass_n", 0)
            by_cat[cat][1] += n
        for cat, (cp, cn) in sorted(by_cat.items()):
            cat_rows.append({
                "model": model,
                "category": cat,
                "passes": cp,
                "total": cn,
                "pass_rate": round(cp / cn, 3) if cn else None,
                "raw_eval_name": raw_name,
            })
    return pd.DataFrame(headline_rows), pd.DataFrame(cat_rows)


# ============================================================
# Sheet 5 — voice metrics (hardcoded from deck slide_voice_gate)
# ============================================================

VOICE_ROWS = [
    # model, avg_chars, bullets_per_resp, bolds_per_resp, emojis_per_resp, boilerplate_opener_pct, voice_profile
    ("Qwen 2.5 7B Instruct (stock)",            672, 1.5,  0.8,  0.4,  62.0, "default chatty"),
    ("Qwen 3 30B-A3B Instruct-2507 (stock)",    335, 1.65, 1.65, 0.05, 68.0, "headline-winning, voice off"),
    ("Skippy 7B v4 ★ PRODUCTION",                157, 0.05, 0.07, 0.02, 5.0,  "target voice — terse, no boilerplate"),
    ("Skippy 14B v4",                            157, 0.07, 0.05, 0.02, 5.0,  "voice transferred up the size axis"),
    ("Skippy MoE v4 (attention-only)",           131, 0.04, 0.04, 0.0,  3.0,  "voice transferred across architectures"),
    ("Skippy MoE v4 + router",                   141, 0.05, 0.05, 0.02, 4.0,  "voice held even with capability recovery"),
]
VOICE_COLS = ["model", "avg_chars", "bullets_per_resp", "bolds_per_resp",
              "emojis_per_resp", "boilerplate_opener_pct", "voice_profile"]


# ============================================================
# Sheet 6 — RTX 5090 performance measurements
# ============================================================
# Source: backend's bakeoff_llm_anchors.py harness (keyhole-sizer commit 4996470).
# Skippy fine-tunes inherit from their base via measurement_alias since FT
# preserves base architecture + GGUF size + compute graph.

PERF_ROWS = [
    # model, gguf_gb, prefill_tok_s_at_2k, decode_tok_s_n256, decode_tok_s_rag_8k,
    # rag_total_sec, source
    ("Qwen 2.5 7B Q4_K_M",        4.68, 10865, 211.7, 183.9, 12.3,  "anchor: qwen2.5-7b-dense"),
    ("Mistral 7B v0.3 Q4_K_M",    4.37, 10217, 239.4, 182.7, 12.6,  "anchor: mistral_7b_v03_dense"),
    ("Llama-3.1 8B Q4_K_M",       4.92, 10162, 211.5, 171.0, 13.3,  "anchor: llama_3_1_8b_dense"),
    ("Skippy 7B v4 (= Qwen 7B alias)",      4.68, 10865, 211.7, 183.9, 12.3,  "FT preserves base GGUF size + compute graph"),
    ("Skippy Mistral v4 (= Mistral alias)", 4.37, 10217, 239.4, 182.7, 12.6,  "FT preserves base GGUF size + compute graph"),
]
PERF_COLS = ["model", "gguf_gb", "prefill_tok_s_at_2k", "decode_tok_s_n256",
             "decode_tok_s_rag_8k", "rag_total_sec", "source"]


# ============================================================
# Sheet 7 — recipe taxonomy matrix (8 filled cells)
# ============================================================

RECIPE_ROWS = [
    # model, base_arch, base_size, lora_targets, loss_mask, corpus, hyperparams, gates, hw, capability_verdict, headline_pct
    ("Skippy 7B v4 ★",                "dense", "7B",   "attention-only",                       "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 2 ep, lr=2e-4 cosine", "✅✅⚠️", "5090", "+3.1pp",   70.5),
    ("Skippy 14B v4",                  "dense", "14B",  "attention + dense FFN",                 "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 2 ep, lr=2e-4 cosine", "✅✅❌", "5090", "+5.3pp",   72.7),
    ("Skippy 32B v4 (3 ep CONFOUND)",  "dense", "32B",  "attention + dense FFN",                 "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 3 ep ⚠",                "❓✅✅", "H100", "−4.6pp",   63.6),
    ("Skippy 32B v4 CLEAN",            "dense", "32B",  "attention + dense FFN",                 "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 2 ep, lr=2e-4 cosine", "⚠️✅⚠️", "H100", "−4.6pp",   63.6),
    ("Skippy MoE v4 (attention-only)", "MoE",   "30B-A3B", "attention-only",                     "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 2 ep",                  "❌✅✅", "H100", "−9.8pp",   61.4),
    ("Skippy MoE v4 + router",         "MoE",   "30B-A3B", "attention + router (gate.weight)",   "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 2 ep",                  "⚠️✅✅", "H100", "−3.8pp",   67.4),
    ("Skippy MoE v4 + router + experts","MoE",  "30B-A3B", "attention + router + packed experts","assistant-only", "alpaca + 100 refusal", "r=8 packed via target_parameters",  "❌⚠️✅", "H100", "−8.3pp",   62.9),
    ("Skippy Mistral v4 (NEW)",        "dense", "7B",   "attention + dense FFN",                  "assistant-only", "alpaca + 100 refusal", "r=64/α=128, 2 ep, lr=2e-4 cosine", "❌✅✅", "5090", "−3.8pp",   56.8),
]
RECIPE_COLS = ["cell_name", "base_arch_class", "base_size_class",
               "lora_target_set", "loss_masking", "corpus_shape",
               "hyperparameters", "gates_capability_voice_safety",
               "hardware_tier", "delta_vs_base", "headline_pct"]


# ============================================================
# Sheet 8 — methodology notes
# ============================================================

METHODOLOGY_ROWS = [
    ("Methodology version",       "2026-05-08-post-remediation", "Bumped after SK-P0-001 (persona quarantined). See eval/EVAL_SET_CHANGELOG.md."),
    ("Eval prompt set",           "v2 (eval/prompts_v2.json)", "42 active prompts × 3 samples = 126 substring-graded scores per run. (Was 44 × 3 = 132 pre-2026-05-08; persona×2 quarantined as BROKEN_SUBSTRING_INCOMPATIBLE.)"),
    ("Categories",                "9 active + 1 broken-quarantined", "Active: coding, general, multihop, numerical_precision, rag_blog, rag_datasheet, rag_email, reasoning, refusal. Quarantined: persona (substring grader can't capture; voice metric tool measures it instead)."),
    ("Grading method",            "Substring matching", "Each prompt has a list of gold substrings; case-insensitive; match_mode='all' = every substring must appear."),
    ("RAG configuration",         "Hybrid retrieval, top-k", "ChromaDB semantic + BM25 + cross-encoder reranker; top-k chunks injected into prompt."),
    ("RAG corpus",                 "61K+ documents",        "i.MX datasheets (NXP), email archive, blog posts, code, internal notes."),
    ("Inference hardware",         "RTX 5090 (32GB)",       "Local llama-cpp-python, n_ctx=16384, all eval runs."),
    ("Quantization",               "Q4_K_M GGUF",            "Same quant across all candidates for apples-to-apples comparison."),
    ("Training hardware (dense)",  "RTX 5090",               "7B + 14B QLoRA fit; 32B requires H100 cloud rental ($25-35/run)."),
    ("Training hardware (MoE)",    "H100 SXM 80GB",          "Qwen3-30B-A3B QLoRA needed cloud ($15-25/run, ~5 hours)."),
    ("Training cost total",         "~$140 cloud + 4.9 free local hr", "Across all 6+ fine-tune iterations of the v4 campaign."),
    ("Voice gate (separate)",       "eval/voice_metrics.py",  "Length, bullets/resp, bolds/resp, emojis/resp, boilerplate opener rate."),
    ("Safety gate",                 "made_up_peripheral probe","Adversarial prompts about fictional features; pass = correct refusal."),
    ("Schema versioning gotcha",    "general n=3 vs n=6",     "Prompt set was extended mid-campaign; legacy entries have general/3 samples, newer have general/6. Δ-vs-base on raw counts is unaffected; per-category rates may not be directly comparable across eval-set versions."),
    ("Eval source repo",            "personal-ai-framework",  "https://github.com/<user>/personal-ai-framework — eval/run_accuracy_eval.py + eval/compare_accuracy_runs.py."),
    ("Recipe taxonomy doc",         "docs/recipe-taxonomy.md","8-dimensional tuple framework. Two recipes that match on all 8 dims should produce the same outcome."),
    ("White paper",                 "docs/skippy-white-paper.md", "Full narrative: iteration arc, gotchas, verification framework, cost arc, recommendations, cross-family + Mistral falsification, recipe taxonomy refs."),
]
METHODOLOGY_COLS = ["aspect", "value", "note"]


# ============================================================
# Sheet 1 — index (built last so it can describe row counts)
# ============================================================

def make_index_rows(headlines, percat):
    return [
        ("models",            len(MODELS_ROWS),    "Every base + fine-tune we evaluated. Family / arch / size / GGUF / training cost / role."),
        ("eval_headlines",    len(headlines),      "One row per eval run (acc_*.json). Pass rate, sample count, timestamp, raw JSON name."),
        ("eval_per_category", len(percat),         "Long format: (model × category) → pass / total / rate. Sums across categories per model = headline."),
        ("voice_metrics",     len(VOICE_ROWS),     "Voice gate measurements. Compare stock cadence (verbose) vs Skippy fine-tune voice (terse)."),
        ("perf_5090",         len(PERF_ROWS),      "RTX 5090 throughput measurements (decode tok/s, prefill, RAG total)."),
        ("recipe_matrix",     len(RECIPE_ROWS),    "8 filled cells in the recipe taxonomy. Dimensions: arch / size / LoRA targets / loss / corpus / hyperparams / gates / hardware."),
        ("methodology",       len(METHODOLOGY_ROWS), "Eval setup, RAG config, hardware, gotchas. Read this first if you're new to the project."),
    ]

INDEX_COLS = ["sheet_name", "row_count", "what_it_contains"]


# ============================================================
# Build
# ============================================================

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    headlines_df, percat_df = scrape_eval_jsons()
    models_df       = pd.DataFrame(MODELS_ROWS, columns=MODELS_COLS)
    voice_df        = pd.DataFrame(VOICE_ROWS, columns=VOICE_COLS)
    perf_df         = pd.DataFrame(PERF_ROWS, columns=PERF_COLS)
    recipe_df       = pd.DataFrame(RECIPE_ROWS, columns=RECIPE_COLS)
    method_df       = pd.DataFrame(METHODOLOGY_ROWS, columns=METHODOLOGY_COLS)
    index_df        = pd.DataFrame(make_index_rows(headlines_df, percat_df), columns=INDEX_COLS)

    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        index_df.to_excel(xl, sheet_name="index", index=False)
        models_df.to_excel(xl, sheet_name="models", index=False)
        headlines_df.to_excel(xl, sheet_name="eval_headlines", index=False)
        percat_df.to_excel(xl, sheet_name="eval_per_category", index=False)
        voice_df.to_excel(xl, sheet_name="voice_metrics", index=False)
        perf_df.to_excel(xl, sheet_name="perf_5090", index=False)
        recipe_df.to_excel(xl, sheet_name="recipe_matrix", index=False)
        method_df.to_excel(xl, sheet_name="methodology", index=False)

    print(f"[build_data_bundle] wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    print(f"  index:           {len(index_df):3d} rows")
    print(f"  models:          {len(models_df):3d} rows")
    print(f"  eval_headlines:  {len(headlines_df):3d} rows")
    print(f"  eval_per_category: {len(percat_df):3d} rows")
    print(f"  voice_metrics:   {len(voice_df):3d} rows")
    print(f"  perf_5090:       {len(perf_df):3d} rows")
    print(f"  recipe_matrix:   {len(recipe_df):3d} rows")
    print(f"  methodology:     {len(method_df):3d} rows")


if __name__ == "__main__":
    main()
