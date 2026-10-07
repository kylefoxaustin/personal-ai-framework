#!/usr/bin/env python3
"""Measure the RETRIEVAL/orchestration cost on this host — the CPU-bound layer the
decode ladder does not capture. Same code runs on the 5090 host and on Thor's ARM,
so the ratio answers: is the agentic orchestration layer slower on edge ARM cores?

Measures all-MiniLM-L6-v2 (Skippy's embedder) query-embed time on CPU and GPU, plus a
cosine search over the real corpus size (31,904 x 384). Reports stable means (n=10).

Run:  <venv>/python3 scripts/agentic_bench/retrieval_proxy.py [path/to/chroma.sqlite3]
"""
import sys, time, sqlite3, statistics
import numpy as np

N_CHUNKS, DIM, REPS = 31904, 384, 10
QUERY = "What does the i.MX 95 Neutron NPU do, and what are its key features?"

def timed(fn, reps=REPS):
    fn()  # warm
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.mean(ts), 2), round(statistics.pstdev(ts), 2)

def load_corpus(path):
    """Best-effort real embeddings from chroma.sqlite3; else a value-independent proxy
    (search time depends on shape, not values)."""
    try:
        con = sqlite3.connect(path); cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings"); n = cur.fetchone()[0]
        con.close()
        print(f"  corpus: chroma.sqlite3 reports {n} embeddings (vectors live in the HNSW "
              f"index, not sqlite rows) -> using a {N_CHUNKS}x{DIM} proxy for search timing")
    except Exception as e:
        print(f"  corpus: sqlite probe failed ({e}); proxy matrix")
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_CHUNKS, DIM), dtype=np.float32)

def main():
    kb = sys.argv[1] if len(sys.argv) > 1 else None
    from sentence_transformers import SentenceTransformer
    import torch
    print(f"host: torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    corpus = load_corpus(kb) if kb else np.random.default_rng(0).standard_normal((N_CHUNKS, DIM), np.float32)
    corpus /= (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)

    m_cpu = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cpu_mean, cpu_sd = timed(lambda: m_cpu.encode([QUERY], show_progress_bar=False))
    print(f"CPU query-embed (all-MiniLM): {cpu_mean} +/- {cpu_sd} ms")

    gpu_mean = None
    if torch.cuda.is_available():
        m_gpu = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        gpu_mean, gpu_sd = timed(lambda: m_gpu.encode([QUERY], show_progress_bar=False))
        print(f"GPU query-embed (all-MiniLM): {gpu_mean} +/- {gpu_sd} ms")

    q = m_cpu.encode([QUERY], show_progress_bar=False)[0]; q /= np.linalg.norm(q) + 1e-9
    def search(): _ = np.argpartition(corpus @ q, -8)[-8:]
    s_mean, s_sd = timed(search)
    print(f"cosine search over {N_CHUNKS}x{DIM}: {s_mean} +/- {s_sd} ms")
    print(f"==> RETRIEVAL cost (CPU embed + search): {round(cpu_mean + s_mean,2)} ms"
          + (f"  | GPU-embed path: {round(gpu_mean + s_mean,2)} ms" if gpu_mean else ""))

if __name__ == "__main__":
    main()
