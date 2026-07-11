#!/usr/bin/env python3
"""Skippy eval corpus — portable, board-agnostic runner.

Stdlib only. No torch, no transformers, no CUDA, no llama.cpp, no ChromaDB,
no network. Runs on a sealed Yocto board with nothing but python3.

The corpus is split into two phases so that *generation* can happen in
whatever runtime the target board has (QNN/QAIRT, TensorRT, llama.cpp,
vLLM, transformers) while *prompt construction* and *scoring* stay
byte-identical to Skippy's reference implementation on the RTX 5090.

    Phase 1  build   →  prompt_pack.jsonl   (fully rendered prompts)
    Phase 2  (yours) →  completions.jsonl   (your runtime generates)
    Phase 3  score   →  acc_<name>.json     (Skippy result schema)

Why the split: the IQ-9075 has no tokenizer library and no chat-template
machinery, so `build` emits the system prompt and the fully-assembled user
message as plain text. Applying the model's own chat template is the
target runtime's job — it is the one thing that legitimately differs per
model, and getting it wrong silently destroys accuracy (see the Qwen3 GGUF
`{% generation %}` incident in Skippy's notes).

RAG WITHOUT A VECTOR DB
-----------------------
`rag_chunks_for_v2.json` is a frozen, pre-retrieved chunk cache: the top
k=8 chunks that Skippy's hybrid retriever (BM25 + semantic + rerank)
returned for each prompt against the 6,731-document knowledge base. Ship
it and you reproduce the exact RAG context without standing up ChromaDB
or moving the corpus. Retrieval is therefore held CONSTANT across boards —
what varies is only the model's ability to use the context. That is
deliberate: it isolates the model from the retriever.

USAGE
-----
    # 1. render the prompts (RAG on, matching Skippy's -v2-rag runs)
    python3 skippy_eval.py build --rag --samples 3 -o prompt_pack.jsonl

    # 2. generate with YOUR runtime, appending one JSON object per line:
    #      {"uid": "<uid from prompt_pack>", "text": "<model output>"}
    #    optional per-sample telemetry: ttft_ms, decode_tok_s, tokens

    # 3. score
    python3 skippy_eval.py score completions.jsonl \
        --name iq9-qwen3-30b-a3b-int8 -o acc_iq9_moe_int8.json

Substring scoring is exact and offline. Skippy's tertiary gates (LLM-judge,
semantic regrade) need cloud API keys and run back on Skippy — send the
acc_*.json home and they get applied there. Do not hand-grade.

IMPORTANT — substring grading has a known family bias. Skippy's own
`regrade_semantic.py` exists because substring matching over-credits models
whose surface phrasing happens to match the gold strings (a Qwen-family
artifact worth up to ~8pp). Treat the substring pass_rate here as a
*screening* number, not a headline. Send the JSON back for semantic regrade
before anyone quotes it.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Byte-identical to eval/run_accuracy_eval_vllm.py and the deployed server.
SYSTEM_PROMPT = (
    "Your name is Skippy. You are a helpful AI assistant. Be direct, "
    "concise, and specific."
)

# Prompts excluded from the headline denominator. Two mechanisms:
#   - the explicit `category_status: BROKEN_SUBSTRING_INCOMPATIBLE` flag on a
#     prompt (authoritative; set by eval/repair_eval_set.py)
#   - the legacy `persona` category check, kept for v2.0 files that predate the flag
# `persona` is structurally incompatible with substring grading (empty
# gold_substrings; the system prompt injects "Skippy" into every model, so it
# cannot differentiate). `general_embedded_book` likewise — a book recommendation
# has no gold. See EVAL_SET_CHANGELOG.md, SK-P0-001.
QUARANTINED_CATEGORIES = {"persona"}
QUARANTINE_FLAG = "BROKEN_SUBSTRING_INCOMPATIBLE"

METHODOLOGY_VERSION = "2026-07-09-eval-set-v2.1-repaired"


def is_quarantined(p: dict) -> bool:
    return (p.get("category_status") == QUARANTINE_FLAG
            or p.get("category") in QUARANTINED_CATEGORIES)


def load_prompts(path: Path) -> list:
    d = json.loads(path.read_text())
    return d["prompts"] if isinstance(d, dict) else d


def build_user_message(user_message: str, rag_chunks: list | None) -> str:
    """Assemble the user turn. Mirrors build_prompt() in
    run_accuracy_eval_vllm.py exactly, minus the chat-template step."""
    if not rag_chunks:
        return user_message.strip()
    parts = []
    for i, c in enumerate(rag_chunks):
        text = c.get("text", "") if isinstance(c, dict) else str(c)
        if not text:
            continue
        src = (c.get("source") or "unknown") if isinstance(c, dict) else None
        header = f"— Excerpt {i+1}" + (f" (source: {src})" if src else "") + " —"
        parts.append(f"{header}\n{text.strip()}")
    chunk_block = "\n\n".join(parts)
    return (
        "Excerpts retrieved from your knowledge base follow. Use them to answer.\n\n"
        f"{chunk_block}\n\n---\n\nQuestion: {user_message.strip()}"
    )


def score(response_text: str, gold_substrings: list, match_mode: str = "all") -> str:
    """Canonical scorer. Copied verbatim from Skippy's runners — do not 'improve' it.

    match_mode="all": every substring must appear (case-insensitive).
    match_mode="any": at least one must appear. Used for refusal prompts,
                      where many phrasings of "I don't know" are valid.
    """
    if not gold_substrings:
        return "manual"
    lower = response_text.lower()
    if match_mode == "any":
        for g in gold_substrings:
            if g.lower() in lower:
                return "pass"
        return "fail (no refusal phrase matched)"
    missing = [g for g in gold_substrings if g.lower() not in lower]
    if not missing:
        return "pass"
    return f"fail (missing: {', '.join(missing)})"


def cmd_build(args):
    prompts = load_prompts(HERE / args.prompts)
    chunks_by_id = {}
    if args.rag:
        rag = json.loads((HERE / args.rag_chunks).read_text())
        chunks_by_id = rag.get("chunks_by_id", {})
        print(f"▶ RAG on: {len(chunks_by_id)} cached chunk sets (k={rag.get('k')})",
              file=sys.stderr)

    n = 0
    with open(args.output, "w") as f:
        for p in prompts:
            chunks = chunks_by_id.get(p["id"]) if args.rag else None
            user = build_user_message(p["prompt"], chunks)
            for s in range(args.samples):
                f.write(json.dumps({
                    "uid": f"{p['id']}#{s}",
                    "prompt_id": p["id"],
                    "sample_index": s,
                    "category": p["category"],
                    "system": SYSTEM_PROMPT,
                    "user": user,
                    # Generation params used for every Skippy headline number.
                    "gen": {"temperature": 0.0, "max_tokens": args.max_tokens},
                }) + "\n")
                n += 1
    print(f"✓ wrote {n} prompts ({len(prompts)} × {args.samples} samples) → {args.output}",
          file=sys.stderr)
    print("  Apply YOUR model's chat template to system+user. Temperature MUST be 0.0 —",
          file=sys.stderr)
    print("  Skippy fine-tunes drop ~26pp at temp=0.3 (grader artifact, not capability).",
          file=sys.stderr)


def cmd_score(args):
    prompts = {p["id"]: p for p in load_prompts(HERE / args.prompts)}

    completions = {}
    with open(args.completions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            completions[c["uid"]] = c

    out_prompts, total, passed, quarantined = [], 0, 0, 0
    for pid, p in prompts.items():
        samples = []
        for uid, c in sorted(completions.items()):
            if c.get("prompt_id") != pid and not uid.startswith(pid + "#"):
                continue
            st = score(c["text"], p.get("gold_substrings", []), p.get("match_mode", "all"))
            samples.append({
                "sample_index": c.get("sample_index"),
                "text": c["text"],
                "status": st,
                "tokens": c.get("tokens"),
                "telemetry": {k: c[k] for k in ("ttft_ms", "decode_tok_s", "elapsed_s")
                              if k in c},
            })
        if not samples:
            continue
        pass_n = sum(1 for s in samples if s["status"] == "pass")
        out_prompts.append({
            **{k: p[k] for k in ("id", "category", "prompt") if k in p},
            "gold_substrings": p.get("gold_substrings", []),
            "match_mode": p.get("match_mode", "all"),
            "samples": samples,
            "aggregate": {
                "pass_n": pass_n,
                "fail_n": sum(1 for s in samples if s["status"].startswith("fail")),
                "manual_n": sum(1 for s in samples if s["status"] == "manual"),
                "n": len(samples),
            },
        })
        if is_quarantined(p):
            quarantined += len(samples)
            continue
        total += len(samples)
        passed += pass_n

    result = {
        "name": args.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompts_version": "2.1",
        "methodology_version": METHODOLOGY_VERSION,
        "samples_per_prompt": args.samples,
        "config": {
            "use_rag": args.rag,
            "temperature": 0.0,
            "grading": "substring",
            "note": "Screening number. Send home for semantic regrade before quoting.",
        },
        "hardware": args.hardware,
        "runtime": args.runtime,
        "prompts": out_prompts,
        "summary": {
            "total_samples": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else None,
            "quarantined_samples": quarantined,
            "quarantined_categories": sorted(QUARANTINED_CATEGORIES),
            "quarantine_flag": QUARANTINE_FLAG,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    pr = result["summary"]["pass_rate"]
    print(f"✓ {args.name}: {passed}/{total} = {pr:.1%} (substring)" if pr is not None
          else f"✓ {args.name}: no scorable samples", file=sys.stderr)
    print(f"  → {args.output}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render prompts → prompt_pack.jsonl")
    b.add_argument("--prompts", default="prompts_v2_1.json")
    b.add_argument("--rag-chunks", default="rag_chunks_for_v2_1.json")
    b.add_argument("--rag", action="store_true", help="inject cached RAG context")
    b.add_argument("--samples", type=int, default=3)
    b.add_argument("--max-tokens", type=int, default=512)
    b.add_argument("-o", "--output", default="prompt_pack.jsonl")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score completions.jsonl → acc_*.json")
    s.add_argument("completions")
    s.add_argument("--prompts", default="prompts_v2_1.json")
    s.add_argument("--name", required=True, help="run label, e.g. iq9-qwen3-moe-int8")
    s.add_argument("--samples", type=int, default=3)
    s.add_argument("--rag", action="store_true")
    s.add_argument("--hardware", default="", help='e.g. "IQ-9075 Hexagon v73"')
    s.add_argument("--runtime", default="", help='e.g. "QAIRT 2.31 context-binary"')
    s.add_argument("-o", "--output", default="acc_run.json")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
