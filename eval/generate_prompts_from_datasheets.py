#!/usr/bin/env python3
"""Generate candidate accuracy-eval prompts from Skippy's datasheet KB.

Workflow:
  1. For each seed topic (memory interfaces, peripherals, cores, etc.), ask
     Skippy to retrieve 1-2 chunks and generate K factual questions about the
     content, each with a distinctive gold substring that should appear in
     a correct answer.
  2. Parse Skippy's output (expected to be JSON lines).
  3. Dedupe + filter for answerability (gold substring non-trivial length).
  4. Emit a JSON list that matches `eval/prompts.json` schema — ready for
     Kyle to spot-check, prune, and merge manually.

Output: eval/results/candidate_prompts_<timestamp>.json

Usage:
  SKIPPY_USER=... SKIPPY_PASSWORD=... \\
      python3 eval/generate_prompts_from_datasheets.py --target 40
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import requests


# Seed topics — each hits RAG to pull chunks from different chapters of
# the datasheets. Covers memory, peripherals, cores, clocks, power, I/O,
# security, packaging, etc.
SEED_TOPICS = [
    "memory interface DRAM LPDDR",
    "UART serial communication peripherals",
    "Arm Cortex cores application processor",
    "I2C SPI peripheral bus",
    "clock speed frequency PLL",
    "power consumption voltage rail supply",
    "GPIO pin multiplexing IOMUX",
    "ethernet MAC PHY connectivity",
    "USB host device interface",
    "PCIe lanes configuration",
    "security trustzone crypto",
    "package dimensions BGA pinout",
    "DMA controller channel",
    "timer counter module peripheral",
    "boot ROM sequence startup",
    "thermal junction temperature TDP",
    "flash memory NAND NOR interface",
    "display LCD MIPI interface",
    "audio I2S SAI peripheral",
    "eIQ Neutron NPU AI accelerator",
]


GEN_PROMPT_TEMPLATE = """You are helping build a test set of factual questions about embedded processors.
Use ONLY the excerpts shown to you by the retrieval system below. Do NOT make up facts.

For the topic "{topic}", produce exactly 2 factual questions with verifiable answers.
Each question must:
- Ask about a SPECIFIC number, name, or short phrase found in the excerpts (not vague like "how does X work")
- Have a short, unambiguous answer that could be checked by substring matching
- Be relevant to embedded-systems engineers using NXP i.MX 9x class chips

Output EXACTLY this JSON format, nothing else (no markdown, no prose):

[
  {{"question": "...", "gold": ["substring1", "substring2"]}},
  {{"question": "...", "gold": ["..."]}}
]

Where each `gold` is a list of distinctive substrings that MUST appear in a correct answer
(2-15 chars each, case-insensitive match). Do NOT use generic words like "the" or "is" as gold.
Prefer proper nouns, numbers with units, or technical identifiers.
"""


def login(endpoint: str) -> dict:
    user = os.environ.get("SKIPPY_USER")
    password = os.environ.get("SKIPPY_PASSWORD")
    if not user or not password:
        print("❌ SKIPPY_USER and SKIPPY_PASSWORD must be set.")
        sys.exit(2)
    resp = requests.post(
        f"{endpoint}/auth/login",
        json={"username": user, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def ask_skippy(endpoint: str, headers: dict, topic: str) -> list:
    """Ask Skippy (with RAG on) to produce 2 Q&A for this topic."""
    prompt = GEN_PROMPT_TEMPLATE.format(topic=topic)
    resp = requests.post(
        f"{endpoint}/generate",
        json={
            "prompt": prompt,
            "use_rag": True,
            "rag_k": 3,
            "skip_agent_loop": True,
            "max_tokens": 500,
            "temperature": 0.3,
        },
        headers=headers,
        timeout=300,
    )
    if resp.status_code != 200:
        print(f"  ⚠️  topic '{topic}' HTTP {resp.status_code}: {resp.text[:120]}")
        return []
    text = resp.json().get("text", "")
    # Parse — Skippy may emit JSON with extra prose. Try to slice the first [...]
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        payload = text[start:end]
        items = json.loads(payload)
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict) and "question" in i and "gold" in i]
    except (ValueError, json.JSONDecodeError):
        print(f"  ⚠️  topic '{topic}': failed to parse JSON from response:")
        print(f"     {text[:200]!r}")
    return []


def dedupe_and_filter(items: list) -> list:
    """Drop near-duplicate questions and ones with weak gold substrings."""
    seen_qs = set()
    keep = []
    for it in items:
        q = (it.get("question") or "").strip()
        gold = it.get("gold") or []
        if not q or not gold:
            continue
        # Normalize for dedupe
        norm = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
        if norm in seen_qs:
            continue
        seen_qs.add(norm)
        # Filter weak gold — skip if any gold substring is <= 2 chars or a stop word
        bad = {"the", "and", "a", "is", "an", "of", "to", "in", "it", "on"}
        strong_gold = [g for g in gold if isinstance(g, str) and len(g) >= 3 and g.lower() not in bad]
        if not strong_gold:
            continue
        keep.append({"question": q, "gold": strong_gold})
    return keep


# Hand-crafted prompts that can't be reliably auto-generated.
HAND_CRAFTED_PROMPTS = [
    # Refusal / anti-hallucination — model should decline, not invent
    {
        "id": "refusal_fictional_chip",
        "category": "refusal",
        "prompt": "What's the maximum clock frequency of the NXP Flux-9000 processor?",
        "gold_substrings": [],  # manual scoring — correct response is a refusal, not a number
        "notes": "manual scoring — the model should say it can't find info, not invent a number",
    },
    {
        "id": "refusal_made_up_peripheral",
        "category": "refusal",
        "prompt": "How many dedicated AI reasoning cores does the i.MX 93's built-in QuantumFlow Engine include?",
        "gold_substrings": [],
        "notes": "manual scoring — 'QuantumFlow Engine' doesn't exist; correct answer refuses",
    },
    {
        "id": "refusal_out_of_scope",
        "category": "refusal",
        "prompt": "Based on the datasheets in your knowledge base, what was the 2029 revenue of NXP Semiconductors?",
        "gold_substrings": [],
        "notes": "manual scoring — datasheets don't contain financial data; model should say so",
    },
    # Multi-hop — requires synthesizing across chunks or documents
    {
        "id": "multihop_imx93_vs_95_cores",
        "category": "multihop",
        "prompt": "Between the i.MX 93 and the i.MX 95, which has more Arm Cortex-A cores, and how many does each have? Give the answer as: '<winner> with N cores vs <loser> with M cores'.",
        "gold_substrings": ["i.MX 95", "i.MX 93"],
    },
    {
        "id": "multihop_peripheral_count",
        "category": "multihop",
        "prompt": "How many I2C instances and how many LPUART instances does the i.MX 93 include, and what's the sum of the two?",
        "gold_substrings": ["LPUART", "I2C"],
    },
    # Numerical precision — exact numbers required
    {
        "id": "numprec_lpddr_rate",
        "category": "numerical_precision",
        "prompt": "At what peak data rate (in MT/s or GT/s) does the i.MX 93 support LPDDR4?",
        "gold_substrings": ["LPDDR4"],
    },
    {
        "id": "numprec_mcu_boot_rom",
        "category": "numerical_precision",
        "prompt": "What is the size of the i.MX 93's internal boot ROM?",
        "gold_substrings": ["KB"],
    },
    # Persona / voice — reuses existing intent
    {
        "id": "persona_brief_greet",
        "category": "persona",
        "prompt": "In one short sentence: how would you describe your own role to a new user?",
        "gold_substrings": [],
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--target", type=int, default=40,
                        help="Target total candidate prompts (default 40)")
    parser.add_argument("--topics", type=int, default=20,
                        help="Number of seed topics to query (default 20)")
    args = parser.parse_args()

    headers = login(args.endpoint)
    topics = SEED_TOPICS[: args.topics]

    print(f"▶ Querying {len(topics)} seed topics, expecting ~2 prompts each...")
    raw_items: list = []
    for t in topics:
        print(f"  · {t}")
        items = ask_skippy(args.endpoint, headers, t)
        for it in items:
            it["_topic"] = t
            it["category"] = "rag_datasheet"
        raw_items.extend(items)
        time.sleep(0.5)  # don't slam the server

    print(f"\n▶ Dedupe + filter (before: {len(raw_items)})...")
    filtered = dedupe_and_filter(raw_items)
    print(f"  kept: {len(filtered)}")

    # Attach id + category to auto-generated items
    auto_prompts = []
    for i, it in enumerate(filtered):
        auto_prompts.append({
            "id": f"rag_datasheet_auto_{i:03d}",
            "category": "rag_datasheet",
            "prompt": it["question"],
            "gold_substrings": it["gold"],
            "source_topic": it.get("_topic"),
        })

    all_candidates = auto_prompts + HAND_CRAFTED_PROMPTS

    out = {
        "version": "candidate-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "auto_from_datasheets": len(auto_prompts),
            "hand_crafted": len(HAND_CRAFTED_PROMPTS),
            "total": len(all_candidates),
        },
        "notes": (
            "Kyle spot-check: prune any auto-generated prompt whose gold "
            "substrings are too generic or whose question doesn't have a "
            "verifiable datasheet answer. Hand-crafted refusal prompts have "
            "empty gold — they need manual scoring (refusal detection)."
        ),
        "prompts": all_candidates,
    }

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"candidate_prompts_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n📄 {out_path}")
    print(f"   {out['source']['total']} candidate prompts ready for spot-check.")


if __name__ == "__main__":
    main()
