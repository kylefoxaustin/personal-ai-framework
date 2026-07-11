#!/usr/bin/env python3
"""Pre-fetch RAG chunks for each prompt in a prompt set — lets us later
run a non-Skippy LLM (e.g. vLLM-served FP8) against the SAME retrieved
context that Skippy would have seen.

Usage (Skippy must be running):
  SKIPPY_USER=... SKIPPY_PASSWORD=... python3 eval/gather_rag_chunks.py \\
      --prompts eval/prompts_v2.json \\
      --out eval/results/rag_chunks_for_v2.json \\
      --k 8
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def login(endpoint: str) -> dict:
    user = os.environ.get("SKIPPY_USER")
    password = os.environ.get("SKIPPY_PASSWORD")
    if not user or not password:
        print("❌ SKIPPY_USER and SKIPPY_PASSWORD must be set.")
        sys.exit(2)
    r = requests.post(f"{endpoint}/auth/login",
                      json={"username": user, "password": password},
                      timeout=15)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['token']}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--rag-categories", default="rag_datasheet,rag_email,rag_blog,multihop,numerical_precision,refusal",
                        help="Only fetch chunks for prompts in these categories (comma-sep). Others get empty chunks.")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompt_set = json.load(f)

    rag_cats = set(c.strip() for c in args.rag_categories.split(",") if c.strip())
    headers = login(args.endpoint)

    out = {"source_prompt_set": args.prompts, "k": args.k, "chunks_by_id": {}}
    for p in prompt_set["prompts"]:
        if p.get("category") not in rag_cats:
            out["chunks_by_id"][p["id"]] = []
            continue
        print(f"▶ {p['id']} [{p['category']}]")
        r = requests.post(f"{args.endpoint}/search",
                          json={"query": p["prompt"], "k": args.k},
                          headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"  ❌ {r.status_code}: {r.text[:120]}")
            out["chunks_by_id"][p["id"]] = []
            continue
        data = r.json()
        results = data.get("results", [])
        # Normalize — different RAG implementations return different shapes.
        chunks = []
        for it in results:
            if isinstance(it, str):
                chunks.append({"text": it, "source": None})
            elif isinstance(it, dict):
                chunks.append({
                    "text": it.get("document") or it.get("text") or it.get("content") or "",
                    "source": it.get("source_file") or it.get("metadata", {}).get("source") or None,
                    "score": it.get("distance") or it.get("score") or None,
                })
        out["chunks_by_id"][p["id"]] = chunks
        print(f"  got {len(chunks)} chunks")
        time.sleep(0.2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n📄 {args.out}")
    print(f"   {sum(1 for v in out['chunks_by_id'].values() if v)} prompts have chunks")


if __name__ == "__main__":
    main()
