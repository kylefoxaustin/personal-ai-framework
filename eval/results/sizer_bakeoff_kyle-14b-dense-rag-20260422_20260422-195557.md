# Sizer bake-off: kyle-14b-dense-rag-20260422

- Timestamp: 2026-04-22T19:55:57
- Endpoint: http://localhost:8080
- Samples per prompt: 5
- Model reported by server: kyle-14b-v3-q4_k_m

## Per-profile summary

| Profile | n | prefill ms (p50/p90) | decode ms (p50/p90) | host ms (p50) | prefill tok/s (p50) | decode tok/s (p50) |
|---|---|---|---|---|---|---|
| rag_qa | 30/30 | 19/23 | 11985/14658 | 527.7 | 2599.8 | 41.4 |

## Per-profile details

### rag_qa

_Retrieval-augmented Q&A over Skippy's 61K-doc knowledge base. Dominated by long prefill (retrieved chunks). Measures: prefill throughput + end-to-end with retrieval overhead._

- Target: ~4800 prefill / ~400 decode tokens
- Critical metric: **end_to_end_latency_with_retrieval**

- **host_ms**: p50=527.69 p90=544.24 p95=552.87 (min=516.57 max=12690.76 mean=936.72 n=30)
- **prefill_ms**: p50=18.71 p90=22.77 p95=1268.48 (min=17.61 max=1307.48 mean=121.6 n=30)
- **decode_ms**: p50=11985.42 p90=14657.63 p95=14763.42 (min=173.41 max=15046.06 mean=11580.92 n=30)
- **total_ms**: p50=12690.35 p90=15300.12 p95=15591.3 (min=727.45 max=24481.93 mean=12639.81 n=30)
- **prefill_tok_per_s**: p50=2599.8 p90=2865.2 p95=4474.7 (min=1888.6 max=5226.1 mean=2741.47 n=30)
- **decode_tok_per_s**: p50=41.4 p90=106.4 p95=116.0 (min=33.3 max=126.9 mean=49.53 n=30)
- **prompt_tokens**: p50=49 p90=57 p95=5676 (min=42 max=5676 mean=518.7 n=30)
- **completion_tokens**: p50=501 p90=502 p95=502 (min=22 max=502 mean=485.2 n=30)
