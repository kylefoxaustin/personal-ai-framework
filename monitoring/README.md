# Skippy Monitoring

Prometheus scrape config + Grafana dashboard for the Personal AI Framework.

## Metrics endpoint

FastAPI server exposes Prometheus text-format metrics at `GET /metrics` on port 8080.

```
curl http://localhost:8080/metrics
```

## Exposed series

| Metric | Type | Labels | What it tracks |
|---|---|---|---|
| `skippy_generations_total` | counter | `endpoint` (`stream`/`generate`) | Total generation requests |
| `skippy_generation_latency_seconds` | histogram | `endpoint` | Wall-clock latency end-to-end |
| `skippy_generation_ttft_seconds` | histogram | — | Time to first token (streaming only) |
| `skippy_generation_tokens_per_second` | histogram | — | Throughput |
| `skippy_tool_calls_total` | counter | `tool`, `outcome` (`executed`/`pending`/`error`) | Agent tool use |
| `skippy_feedback_total` | counter | `rating` (`up`/`down`) | 👍/👎 clicks |
| `skippy_reminders_created_total` | counter | — | Reminders scheduled |
| `skippy_reminders_fired_total` | counter | — | Reminders that surfaced in the UI |
| `skippy_rag_queries_total` | counter | `mode` (`standard`/`synthesis`) | RAG retrievals |
| `skippy_rag_docs_returned` | histogram | — | Chunks returned per retrieval |
| `skippy_model_loaded` | gauge | — | `1` if LLM is loaded |
| `skippy_maintenance_mode` | gauge | — | `1` if retraining |
| `skippy_knowledge_base_docs` | gauge | — | Doc count in ChromaDB |

## Running Prometheus + Grafana locally

Quickest way — two containers on the same network as Skippy:

```bash
# from repo root
docker run -d --name skippy-prom \
  --add-host=host.docker.internal:host-gateway \
  -p 9090:9090 \
  -v "$PWD/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus

docker run -d --name skippy-grafana -p 3001:3000 grafana/grafana
```

Then:
1. Visit http://localhost:3001 (admin / admin).
2. Add a Prometheus data source → URL `http://host.docker.internal:9090`.
3. Dashboards → **Import** → upload `monitoring/grafana-dashboard.json`.

## Notes

- The dashboard targets a single Skippy instance. For multi-host, add a `host` label to the scrape job.
- Histogram buckets are tuned for a single-GPU workstation running Qwen 2.5 14B (TTFT ≤ ~2s, end-to-end ≤ ~60s). Adjust `pipeline/metrics.py` if you run a larger model.
