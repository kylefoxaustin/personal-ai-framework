"""
Run a distribution benchmark against local Skippy for the Keyhole sizer cross-reference.
Drives /generate/stream across 5 workload categories and collects per-call TTFT + tok/s,
then emits p50/p95 per category.
"""
import json, time, statistics, subprocess, urllib.request, urllib.error, sys

BASE = "http://localhost:8080"
USER = "kyle"
PASSWORD = "123456"


def login():
    req = urllib.request.Request(
        f"{BASE}/auth/login",
        data=json.dumps({"username": USER, "password": PASSWORD}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    return d["token"]


def stream(token, prompt, max_tokens=256, context=None, use_rag=False):
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "use_rag": use_rag,
    }
    if context:
        body["context"] = context
    req = urllib.request.Request(
        f"{BASE}/generate/stream",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    r = urllib.request.urlopen(req, timeout=600)
    metrics = None
    for raw in r:
        line = raw.decode(errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "done":
            metrics = ev.get("metrics", {})
            break
    return metrics


def p(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, int(round((q / 100) * (len(s) - 1))))
    return s[i]


def run_category(token, name, prompts, **kw):
    print(f"\n=== {name} ({len(prompts)} prompts) ===")
    ttfts, tps = [], []
    for i, p_ in enumerate(prompts, 1):
        t0 = time.time()
        m = stream(token, p_, **kw)
        wall = time.time() - t0
        if not m:
            print(f"  [{i}] no metrics returned (wall {wall:.2f}s)")
            continue
        ttfts.append(m["ttft"])
        tps.append(m["throughput"])
        print(f"  [{i}] ttft={m['ttft']:.3f}s  tok/s={m['throughput']:.1f}  "
              f"tokens={m['tokens']}  wall={wall:.1f}s")
    return {
        "n": len(ttfts),
        "ttft_p50": p(ttfts, 50), "ttft_p95": p(ttfts, 95),
        "tps_p50":  p(tps,   50), "tps_p95":  p(tps,   95),
        "tps_min":  min(tps) if tps else None, "tps_max": max(tps) if tps else None,
        "raw_tps":  tps,
    }


def wait_ready():
    """Block until /health says model_loaded, up to 3 minutes."""
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
                d = json.loads(r.read())
                if d.get("model_loaded"):
                    return True
        except (urllib.error.URLError, ConnectionResetError):
            pass
        time.sleep(1)
    return False


def main():
    results = {}

    # ---------- (a) cold-start ----------
    print("=== restarting llm-server for true cold-start ===")
    subprocess.run(["docker", "compose", "-f",
                    "/home/kyle/Documents/GitHub/personal-ai-framework/docker-compose.yaml",
                    "restart", "llm-server"],
                   check=True, capture_output=True)
    if not wait_ready():
        print("server didn't come up", file=sys.stderr); sys.exit(1)
    time.sleep(3)  # let chat template load too
    token = login()
    print("ready; firing cold-start prompt")
    m = stream(token, "Hello. Give me a one-sentence greeting.", max_tokens=64)
    if m:
        print(f"  cold-start: ttft={m['ttft']:.3f}s  tok/s={m['throughput']:.1f}  tokens={m['tokens']}")
        results["cold_start"] = {
            "n": 1, "ttft_p50": m["ttft"], "ttft_p95": m["ttft"],
            "tps_p50": m["throughput"], "tps_p95": m["throughput"],
            "tps_min": m["throughput"], "tps_max": m["throughput"],
            "raw_tps": [m["throughput"]],
        }

    # ---------- (b) plain chat ----------
    plain = [
        "What's a fun fact about the moon?",
        "Write a haiku about debugging.",
        "Summarize ARM Cortex-M vs Cortex-A in two sentences.",
        "What's the capital of Luxembourg?",
        "Draft a short Slack message saying I'm running late to a meeting.",
    ]
    results["plain_chat"] = run_category(token, "plain_chat", plain, max_tokens=200)

    # ---------- (c) long context (RAG-heavy proxy) ----------
    # Stuff ~5K tokens of synthetic context via the `context` field so decode
    # happens after a real long prefill — models the RAG-heavy case without
    # needing an actual knowledge base ingested.
    long_ctx_blob = " ".join([
        "LPDDR5X is a low-power double-data-rate memory standard widely used in edge inference SoCs.",
        "The 8.4 GT/s variant on a 128-bit bus yields 134.4 GB/s peak bandwidth.",
        "Usable bandwidth at 75 percent utilization is 100.8 GB/s.",
    ] * 200)   # ~4-6K tokens
    long_prompts = [
        "Based on the memory context above, estimate decode tok/s for a 3B-active MoE at Q4_K_M quantization.",
        "Given the context, summarize the bandwidth-versus-compute tradeoff in one paragraph.",
        "From the context, what is the per-token memory bandwidth requirement for a 14B dense Q4 model at 30 tok/s?",
        "Based on that context, compare MoE and dense models for edge inference.",
        "Using the memory facts above, explain why Mixtral 8x7B doesn't fit a 16 GB SKU.",
    ]
    results["rag_heavy"] = run_category(token, "rag_heavy (long context)",
                                        long_prompts, max_tokens=256,
                                        context=[long_ctx_blob])

    # ---------- (d) long-form analytical (reasoning-ish) ----------
    reason = [
        "Walk through, step by step, how you'd decide between Qwen3-30B-A3B MoE and Qwen 2.5 14B dense for an edge NPU with 16 GB of LPDDR5X.",
        "Step by step: design a LoRA training data pipeline that collects conversations from a personal AI, filters low-quality ones, and produces alpaca format.",
        "Reason through: given a 200 TOPS NPU with 100 GB/s memory bandwidth, is the system compute-bound or memory-bound for a 3B-active MoE at Q4_K_M? Show the math.",
        "Explain in detail why a MoE model's TPS scales sub-linearly with compute but near-linearly with memory bandwidth.",
        "Plan out the steps to fine-tune a MoE model on a rented H100, including gotchas.",
    ]
    results["reasoning"] = run_category(token, "reasoning (long-form analytical)",
                                        reason, max_tokens=400)

    # ---------- emit results ----------
    print("\n\n=== SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
