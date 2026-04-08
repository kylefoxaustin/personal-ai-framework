# CLAUDE.md

## Project Overview

Personal AI Framework - a fully local AI assistant with a 61K+ document knowledge base. Runs Mixtral 8x7B (Q4 GGUF) via llama-cpp-python on NVIDIA GPU, with ChromaDB for RAG and SQLite for conversation storage. The AI persona is named "Skippy."

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
- Current version: v5.5.0
- Commit style: `vX.Y.Z: Short Description` for releases, `Update README for vX.Y.Z - Feature Name` for README updates

## Common Tasks

- **Adding a UI feature**: Edit `web/index.html`. Add HTML, CSS (in `<style>`), and JS (in `<script>`) in the same file.
- **Adding a backend endpoint**: Edit `pipeline/llm_server.py`. Add FastAPI route.
- **New version release**: Update README badge + version table, commit, tag, push with `--tags`.
