# CLAUDE.md

## Project Overview

Personal AI Framework - a fully local AI assistant with a large document knowledge base. Runs a fine-tuned Qwen 2.5 7B (Kyle v4, Q4 GGUF — `models/qwen2.5-7b-kyle/kyle-qwen25-7b-v4-q4_k_m.gguf`) via llama-cpp-python on NVIDIA GPU, with ChromaDB for RAG and SQLite for conversation storage. The AI persona is named "Skippy." (Production moved from Qwen 2.5 14B Instruct to the fine-tuned 7B v4 in the v5.10.0 campaign — see README's version table; the actual loaded model is always derived from `pipeline/config.yaml`'s `model.path`.)

## Architecture

- **Web UI** (`web/index.html`): Single-file HTML/CSS/JS app served via `python3 -m http.server 3000`. No build step, no framework.
- **LLM Server** (`pipeline/llm_server.py`): FastAPI server on port 8080. Handles chat, streaming, RAG queries, file uploads, transcription, OCR, email, memory, and settings.
- **Vector DB**: ChromaDB container on port 8000 (`vectordb/` volume).
- **Docker**: Two services defined in `docker-compose.yaml` - `llm-server` (GPU, mounts models/pipeline/knowledge) and `vectordb`.

## Key Directories

```
pipeline/          # Python backend (FastAPI server + all services)
web/               # Single-page web UI (index.html only)
docker/            # Dockerfile + files copied into container
models/            # GGUF model files (not committed)
knowledge/         # Knowledge base documents (mounted into container)
vectordb/          # ChromaDB persistent storage
training/          # LoRA fine-tuning scripts and output
scripts/           # Helper scripts (model download)
docs/              # Guides (architecture, FAQ, quickstart, fresh install)
```

## Web UI Conventions

- Everything is in `web/index.html` - HTML, CSS, and JS in one file.
- The UI calls the FastAPI backend at `http://localhost:8080` directly.
- Major UI panels: Chat (main), Conversation Sidebar, Settings (gear icon), Upload Panel (header button), Memory Viewer.
- Upload Panel (v4.0.0): Dedicated overlay with three tabs - Documents, Audio/Video, Images. Accessed via header button, not Settings.
- Settings Panel contains: AI Personality, Backup & Restore, Web Search toggle, Email Providers, Daily Digest, Auto-Sync.

## Backend Services (pipeline/)

| File | Purpose |
|------|---------|
| `llm_server.py` | FastAPI app - all HTTP endpoints |
| `rag_service.py` | ChromaDB RAG integration |
| `advanced_rag.py` | Hybrid search (BM25 + semantic + reranking) |
| `conversation_store.py` | SQLite conversation persistence |
| `memory_service.py` | Memory ingestion and search |
| `meeting_summarizer.py` | Transcript summarization |
| `doc_generator.py` | Spec/proposal/report generation |
| `email_service.py` | Gmail OAuth integration |
| `daily_digest.py` | Scheduled activity summaries |
| `settings_manager.py` | Settings read/write (`~/.personal-ai/settings.json`) |
| `smart_chunker.py` | Document chunking for ingestion |
| `sync_service.py` | Knowledge base sync |
| `ingest_knowledge.py` | General document ingestion |
| `ingest_datasheets.py` | PDF datasheet ingestion |
| `ingest_pst_emails.py` | PST email import |

## Development Workflow

- Start services: `./run.sh start` (docker compose up + python http server)
- Stop: `./run.sh stop`
- Web UI changes are instant (just reload browser, no build)
- Backend changes require container restart
- Config: `pipeline/config.yaml` (model paths, RAG settings)
- User settings: `~/.personal-ai/settings.json`

## Versioning

- Version badge is in `README.md` (shields.io badge)
- Version history table is at the bottom of `README.md`
- Releases are tagged as `vX.Y.Z` on main branch
- Current version: v5.6.0
- Commit style: `vX.Y.Z: Short Description` for releases, `Update README for vX.Y.Z - Feature Name` for README updates

## Common Tasks

- **Adding a UI feature**: Edit `web/index.html`. Add HTML, CSS (in `<style>`), and JS (in `<script>`) in the same file.
- **Adding a backend endpoint**: Edit `pipeline/llm_server.py`. Add FastAPI route.
- **New version release**: Update README badge + version table, commit, tag, push with `--tags`.

## ratchet ecosystem — Skippy is the upstream author (phase 5, v5.10.1)

`ratchet` is a shared SoC sizing engine consolidated from four ecosystem
surfaces. Skippy's relationship to that ecosystem is **structurally upstream**:
this repo *authors* canonical artifacts that the rest of the ecosystem consumes.
It is **not** a downstream consumer of ratchet and has **no `import ratchet`**
anywhere.

The full empirical recon lives in
[`docs/decisions/phase5-scope-recon.md`](docs/decisions/phase5-scope-recon.md).
Short version for future contributors:

**Skippy authors / produces:**
- [`docs/private_anchor_secrets_spec.md`](docs/private_anchor_secrets_spec.md)
  — the canonical anchor-secrets schema. ratchet's anchor loader, PAI sizer's
  loader, and keyhole-sizer's loader all conform to this spec; this file is
  the source of truth.
- [`eval/build_sizer_bundle.py`](eval/build_sizer_bundle.py) — produces
  `eval/results/sizer_bundle.json`, the bundle PAI sizer's `measured.py`
  consumes to populate `RTX_5090_REFERENCE.measured_llm`.

**Data flow:**
```
  Skippy (this repo)
    ├─ anchor-secrets schema   ──► ratchet (implements) ──► PAI sizer, keyhole-sizer
    └─ sizer_bundle.json       ─────────────────────────► PAI sizer (measured.py)
```

**Why no `requirements.txt` pin on ratchet.** Pinning a dependency Skippy
doesn't import would misrepresent the upstream-producer relationship as a
downstream-consumer one. A contributor auditing `requirements.txt` would see
`ratchet>=…` and grep for `import ratchet`, find nothing, and either think the
dependency is dead code or assume non-obvious dynamic loading — both wrong.
This CLAUDE.md section is the truthful hook for the relationship. (Contrast:
keyhole backend v1.0.1 *does* pin ratchet because it's a sibling that *could*
consume — a future-use hook for a real candidate consumer. Different
relationship, different shape.)

**If you're adding cross-surface schemas or sizer-bundle fields:** update the
spec in `docs/private_anchor_secrets_spec.md` (Skippy is the authority), then
the downstream consumers (ratchet, PAI sizer, keyhole-sizer) update their
loaders/measured.py to match. The discipline is one-way: Skippy → ratchet → sizers.

**Ecosystem checkpoint after phase 5:**

| Surface | Relationship | Tag |
|---|---|---|
| PAI sizer | consumer (Hardware/TIERS/loader/capability) | v1.1.0 |
| keyhole-sizer | consumer (+ vision adapters) | v1.1.0 |
| keyhole backend | sibling, future-use pin | v1.0.1 |
| Skippy (this repo) | upstream author | **v5.10.1** |

