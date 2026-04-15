"""
Prometheus metrics — expose operational counters/histograms/gauges at /metrics.

Keep metric cardinality low (no per-user, per-message IDs as labels). Labels
we DO use: tool name, rating ('up'/'down'), endpoint. Everything else is a
bucketed histogram or a scalar gauge.
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Dedicated registry so we don't pull in process/platform collectors we don't need.
registry = CollectorRegistry()

# ── Generation ──
generations_total = Counter(
    "skippy_generations_total",
    "Total generation requests handled",
    ["endpoint"],  # 'stream' or 'generate'
    registry=registry,
)
generation_latency_seconds = Histogram(
    "skippy_generation_latency_seconds",
    "Wall-clock time for a generation (end to end, including RAG + tools)",
    ["endpoint"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
    registry=registry,
)
generation_ttft_seconds = Histogram(
    "skippy_generation_ttft_seconds",
    "Time to first token (streaming only)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8),
    registry=registry,
)
generation_tokens_per_second = Histogram(
    "skippy_generation_tokens_per_second",
    "Throughput of a generation in tokens/second",
    buckets=(5, 10, 20, 40, 60, 80, 100, 150),
    registry=registry,
)

# ── Tool use ──
tool_calls_total = Counter(
    "skippy_tool_calls_total",
    "Tool invocations, labeled by tool name and outcome",
    ["tool", "outcome"],  # outcome: 'executed', 'pending', 'error'
    registry=registry,
)

# ── Feedback ──
feedback_total = Counter(
    "skippy_feedback_total",
    "User feedback on assistant responses",
    ["rating"],  # 'up' or 'down'
    registry=registry,
)

# ── Reminders ──
reminders_created_total = Counter(
    "skippy_reminders_created_total",
    "Reminders scheduled",
    registry=registry,
)
reminders_fired_total = Counter(
    "skippy_reminders_fired_total",
    "Reminders that reached their due time and were surfaced to the UI",
    registry=registry,
)

# ── RAG ──
rag_queries_total = Counter(
    "skippy_rag_queries_total",
    "RAG retrieval calls",
    ["mode"],  # 'standard' or 'synthesis'
    registry=registry,
)
rag_docs_returned = Histogram(
    "skippy_rag_docs_returned",
    "Number of chunks returned from a RAG retrieval",
    buckets=(0, 1, 3, 5, 8, 12, 20, 40),
    registry=registry,
)

# ── Model / system gauges ──
model_loaded = Gauge(
    "skippy_model_loaded",
    "1 if the LLM is loaded in GPU memory, 0 otherwise",
    registry=registry,
)
maintenance_mode = Gauge(
    "skippy_maintenance_mode",
    "1 if server is in training/retraining mode (generations rejected)",
    registry=registry,
)
knowledge_base_docs = Gauge(
    "skippy_knowledge_base_docs",
    "Number of documents currently in the ChromaDB knowledge base",
    registry=registry,
)


def render_text() -> bytes:
    return generate_latest(registry)
