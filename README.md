# 🧠 Personal AI Framework

A private AI assistant that runs **100% on your own hardware**. It answers from your documents, uses tools (files, web, email, calendar), remembers your conversations, and can be fine-tuned on your own data. The model and your data stay on the machine; web search and Gmail are opt-in.

It is also a **measured reference workload for agentic AI on edge silicon**. The same agent runs a 10-task benchmark from an RTX 5090 down to Jetson, Qualcomm and NXP i.MX boards, and every number it reports carries its provenance.

![Version](https://img.shields.io/badge/version-5.11.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![x86 demo (Windows + Linux)](https://github.com/kylefoxaustin/personal-ai-framework/actions/workflows/x86-demo.yml/badge.svg)](https://github.com/kylefoxaustin/personal-ai-framework/actions/workflows/x86-demo.yml)

## 🔬 What we found

All numbers below are **MEASURED** (we ran them) unless tagged otherwise. Model: Qwen 2.5 7B, Q4_K_M.

**1. Agent speed tracks memory bandwidth, not TOPS.** Decode speed, same 7B model, across six hardware configs:

| Hardware | Decode tok/s | Notes |
|---|--:|---|
| RTX 5090 (desktop) | 217.9 | full Skippy stack, n=3 |
| Jetson AGX Thor | 40.85 | llama-bench |
| Jetson AGX Orin | 27.92 | llama-bench |
| Qualcomm IQ-9075 (Hexagon NPU) | 9.45 | Genie runtime; NPU path caps prompts at ~256 tokens |
| NXP i.MX95 + Kinara ARA240 | 6.3 | July 2026 run, within 3% of NXP's published 6.51 |
| NXP i.MX95, A55 CPU only | 1.85 | llama-bench |

On the GPU and Jetson tiers that works out to **57–70% of each chip's rated memory bandwidth** (DERIVED: tok/s × model size). Moving from 7B to 14B costs **1.9×** on the 5090, Thor and Orin alike, matching the parameter ratio.

**2. Agents pay a tax outside the model.** On the 5090, every document-grounded task spends a **fixed 0.8–1.0 s** in retrieval, prompt building and the agent loop (it repeats to ~1% between runs). That's 15–38% of the task. Projected onto Thor's ARM cores it is ~5 s, about 39% of a RAG task (DERIVED). A cold first call takes **89 s vs ~2 s warm**, almost all of it loading the retrieval stack (n=1).

**3. Fine-tuning made the model more careful.** On a 117-answer faithfulness sweep graded by two independent judges, answers flagged as fabricated:

| Model | Fabricated (of 117) |
|---|--:|
| Old fine-tune v4 (retired) | 36 |
| Stock Qwen 2.5 7B Instruct | 22 |
| **v5-clean fine-tune (production)** | **12** (repeat run: 16) |

**4. Small models still invent specs.** Two judges (Claude Sonnet + GPT-4o) graded the stock 7B on the 10 agentic tasks: they agreed on 4 clean passes (file ops, transcription, OCR, script running). Open-ended writing tasks fabricated hardware details even when fluent, so we report cost per *successful* task, not cost per task.

Full write-ups: [agentic findings](eval/results/ladder/agentic/FINDINGS.md) · [benchmark runbook](docs/RUN_BENCHMARKS.md) · [white paper](docs/skippy-white-paper.md)

## 🧪 What we test

| Suite | What it checks | Run it |
|---|---|---|
| **Agentic benchmark** (10 tasks) | Email drafting, web search + summarize, datasheet Q&A, file read→summarize→write, Whisper transcription, OCR, meeting transcribe→summarize, sandboxed script, multi-tool chain, long-context doc brief. Records wall time, prefill/decode split, host overhead and GPU/CPU load per task. Outputs must be coherent to count. | `python3 scripts/agentic_bench/trace_bench.py` |
| **Full system run** | The benchmark + the board ladder + two-judge grading, rolled into one report. | `python3 scripts/run_benchmark.py` |
| **Board ladder** | Reserves a board, checks the model fits, stages it, runs decode/prefill, writes a standard result. Thor, Orin, i.MX95 (CPU and ARA240), iq9. | `python3 scripts/bench_board.py <board>` |
| **Task success** | Two judges from different model families grade each agentic output (correct / partial / fabricated). Splits are flagged, never averaged. | `python3 eval/judge_agentic_tasks.py` |
| **Faithfulness sweep** | 117 grounded answers at production temperature (0.7), with repeats; two judges check every answer against its source. | `python3 eval/faithfulness_sweep.py gen` |
| **Accuracy eval** | 44 prompts × 3 samples: datasheet RAG, multi-hop, numerical precision, refusal, reasoning, coding, persona. | `python3 eval/run_accuracy_eval.py --name <run> --prompts eval/prompts_v2_1.json --with-rag` |
| **Retrieval** | Recall@k measured on its own, never through the model or a judge; checks whether each ground-truth fact is in the index at all vs not retrieved. | `python3 eval/bench_retrieval.py` · `python3 eval/retrieval_fact_recall.py` |
| **Standard benchmarks** | TruthfulQA, MMLU, HellaSwag against the live server (RAG off by default). | `python3 eval/run_standard_benchmarks.py --name <run>` |
| **Integration tests** (39) | Server health, generation, stop tokens, personality, facts, RAG (product-ID tokenizing, alias routing, persona not hijacked), and the full LoRA retrain cycle: maintenance mode, 503 blocking, recovery, dedup, no fact leakage. | `python3 tests/test_integration.py --phase 1` (`--all` ≈ 2.5 h) |
| **x86 CPU demo, CI** | Every change to the demo runs it on fresh Windows and Linux machines. | badge above |

**How we keep the numbers honest**
- Every number is tagged **MEASURED**, **DERIVED** or **SOURCED**, and the tiers are never compared against each other.
- Anything leaving the repo is graded by **two judges from different model families**. One judge, or a judge reading the same gold strings as a substring check, doesn't count as confirmation.
- Each layer is measured directly. Retrieval is benchmarked on its own because a retriever miss once showed up as a fake "INT8 accuracy loss".
- A broken output is cheaper to compute, so its timing doesn't count.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Conversation Memory** | AI remembers past conversations - learns your name, preferences, context |
| **Memory Viewer** | Browse, search, and delete specific memories |
| **Customizable Personality** | Name your AI, set its personality and traits |
| **Document Upload** | Drag & drop PDFs, TXT, MD files directly in the web UI |
| **Audio/Video Transcription** | Upload MP4, MP3, WAV files - transcribed with Whisper |
| **Meeting Summarization** | Auto-extract key points, action items, decisions from transcripts |
| **Web Search** | Search the internet with "search: query" (toggle in Settings) |
| **Image/Screenshot OCR** | Extract text from images with Tesseract OCR |
| **Export/Import** | Backup and restore all conversations and settings |
| **Personal Knowledge Base** | Index your documents, transcripts, notes and datasheets; tested on a 31,939-chunk datasheet corpus |
| **Your Own Fine-Tune** | LoRA pipeline in `training/` to fine-tune the model on your own data |
| **Performance Metrics** | TTFT, tokens/sec, and total time on every response |
| **Gmail Integration** | OAuth-based Send & Draft buttons - email directly from the UI |
| **Streaming Responses** | Real-time word-by-word output |
| **Hybrid RAG Search** | BM25 + semantic + reranking for best results |
| **Auto-Remember** | Toggle to automatically save conversations to memory |
| **Meeting Summarizer** | Transcribe audio/video with Whisper + summarize |
| **Document Generator** | Create specs, proposals, reports from your knowledge |
| **Datasheet Ingestion** | Ingest PDFs - reference manuals, app notes |
| **Daily Digest** | Automated morning summary of your AI activity |
| **Web UI** | Clean interface with settings panel and conversation sidebar |

## 🖥️ Requirements

- **GPU**: NVIDIA RTX 3080+ (16GB+ VRAM recommended)
- **RAM**: 32GB+
- **Storage**: 50GB+ for models and knowledge base
- **OS**: Ubuntu 22.04+ (or similar Linux) for the full assistant
- **Windows**: the CPU-only x86 demo runs natively (see below). The full assistant (Docker + NVIDIA GPU) is tested on Linux only.

## 🚀 Quick Start
```bash
# Clone the repo
git clone https://github.com/kylefoxaustin/personal-ai-framework.git
cd personal-ai-framework

# Run setup (checks GPU/Docker, installs dependencies, downloads a model, builds images)
#   choose 1) Qwen 2.5 7B Instruct (recommended); setup points pipeline/config.yaml at it
./setup.sh

# Start services (LLM server :8080, vector DB, web UI :3000)
./run.sh start

# Open web UI
xdg-open http://localhost:3000
```

> **About the model.** A fresh install runs **stock Qwen 2.5 7B Instruct**, which is public on Hugging Face. The fine-tuned Skippy model (v5-clean, built on that same base) is **not distributed** and never will be. It stays on the author's machine. To build your own fine-tune on your own data, use the scripts in `training/` (`training/build_grounded_dataset.py`, `training/train_lora_v5_clean.py`). On the faithfulness sweep below, stock 7B fabricated in 22 of 117 answers and v5-clean in 12 (MEASURED), so the fine-tune mainly makes the model more careful.

## 🪟 Try It on Windows (x86 CPU demo)

You don't need a GPU, Docker or the server for this. The demo downloads a small model (Qwen2.5-3B, 2 GB) and benchmarks it on your CPU. It shows that token generation is limited by memory bandwidth, not compute. It also runs three model-only agentic tasks: drafting an email, summarizing a file, and answering spec questions from facts in the prompt.

In PowerShell (Python 3.9+ from python.org or `winget install Python.Python.3.12`):
```powershell
git clone https://github.com/kylefoxaustin/personal-ai-framework.git
cd personal-ai-framework
py -m pip install --prefer-binary llama-cpp-python huggingface-hub --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
py scripts\x86_demo.py --tasks
```
The prebuilt wheel means you don't need a compiler. Use `py`, not `python3`, which on Windows often opens the Microsoft Store. The same demo runs on Linux and WSL2 with `python3 scripts/x86_demo.py`. Every change to the demo runs it on fresh Windows and Linux machines (badge above).

Options, output explained, and troubleshooting: [docs/RUN_X86_DEMO.md](docs/RUN_X86_DEMO.md).

## 🎭 AI Personality

Give your AI a name and personality! Click ⚙️ Settings in the web UI:

- **AI Name**: "Skippy", "Jarvis", "Friday", or whatever you want
- **Personality Prompt**: "You are Skippy, a sarcastic but helpful AI who loves sci-fi references..."

The personality is injected into every prompt automatically.

## 🧠 Conversation Memory

Your AI remembers past conversations:

- **Auto-Remember**: Toggle to automatically save all chats to memory
- **Manual Save**: Click 💾 on any conversation to save it (turns to 🧠 when saved)
- **Memory Viewer**: Click 🧠 in header to browse/delete specific memories
- **Clear Memory**: 🗑️ button to wipe all memories (with confirmation)
- **Memory Search**: AI searches past conversations when answering

Ask "What's my name?" or "What did we discuss about X?" - the AI remembers!

## 📤 Upload Panel

Upload documents, audio/video, and images to your knowledge base from a single dedicated panel:

1. Click **📤 Upload** in the header
2. Choose a tab: **📄 Documents**, **🎤 Audio/Video**, or **📷 Images**
3. Drag & drop files, click to browse, or use **📂 Upload Folder** to select an entire folder
4. Multi-file and folder uploads show a progress bar with per-file results
5. Files are automatically processed and indexed in ChromaDB

### 📄 Documents
- Select document type (General, Datasheet, Transcript, Email)
- **Supported formats**: PDF, TXT, MD

### 🎤 Audio/Video Transcription
- Enter a recording title (optional)
- Toggle meeting summary generation (key points, action items, decisions)
- Whisper transcribes using CPU to avoid VRAM conflicts
- Results show full summary, key points, action items, and decisions
- Copy Summary / Copy Transcript / Send Email / Save Draft buttons
- Collapsible full transcript view
- Transcripts saved to `~/knowledge/transcripts/` and searchable by Skippy!
- **Supported formats**: MP4, MP3, WAV, M4A, WEBM, MKV, MOV, AVI, OGG, FLAC

### 📷 Image/Screenshot OCR
- Enter an image title (optional)
- Optional summarization of extracted text (checkbox)
- Tesseract extracts text and auto-ingests to ChromaDB
- Copy Text / Copy Summary / Send Email / Save Draft buttons
- Perfect for whiteboard photos, screenshots, scanned docs, or book pages!
- **Supported formats**: PNG, JPG, GIF, BMP, TIFF, WEBP

## 💾 Backup & Restore

Never lose your AI's memories:

- **📤 Export Conversations**: Download all chats as JSON
- **💾 Full Backup**: Export conversations + settings in one file
- **📥 Import Backup**: Restore from any backup file

Find these in ⚙️ Settings → Backup & Restore

## 📁 Project Structure
```
personal-ai-framework/
├── docker/                 # Docker configurations
├── knowledge/              # Your knowledge base (mounted volume)
│   ├── documents/          # General documents
│   ├── emails/             # Exported emails (.eml, .mbox)
│   ├── transcripts/        # Meeting/video transcripts
│   ├── datasheets/         # PDF datasheets & manuals
│   └── writing/            # Your blog posts, articles
├── pipeline/               # Core Python services
│   ├── llm_server.py       # FastAPI LLM server
│   ├── rag_service.py      # ChromaDB RAG
│   ├── advanced_rag.py     # Hybrid search + reranking
│   ├── conversation_store.py  # SQLite conversation storage
│   ├── memory_service.py   # Memory ingestion & search
│   ├── meeting_summarizer.py
│   ├── doc_generator.py
│   ├── email_service.py
│   ├── daily_digest.py
│   └── settings_manager.py
├── web/                    # Web UI
├── training/               # LoRA fine-tuning
└── run.sh                  # Main CLI
```

## 🛠️ Commands

### Basic Operations
```bash
./run.sh start              # Start all services
./run.sh stop               # Stop all services
./run.sh status             # Check service status
./run.sh logs               # View logs
```

### Knowledge Base
```bash
./run.sh sync               # Sync knowledge base
./run.sh ingest-datasheets  # Ingest PDF datasheets
./run.sh ingest-emails      # Ingest email exports
```

### AI Features
```bash
# Email drafting
./run.sh email draft "project status update" -t recipient@email.com
./run.sh email draft "follow up on meeting" --open  # Opens in email client

# Document generation
./run.sh generate technical_spec "i.MX display system"
./run.sh generate project_proposal "embedded AI project"
./run.sh generate status_report "Q1 development progress"

# Meeting summarization
./run.sh summarize recording.mp4 -t "Team Meeting" --add-to-kb

# Daily digest
./run.sh digest                          # Print digest
./run.sh digest -m mailto -t you@email.com  # Open in email client
./run.sh digest --schedule 08:00 -t you@email.com  # Schedule daily
```

## 🌐 Web UI

Access at `http://localhost:3000`

### Main Features
- **Conversation Sidebar**: All your chats, with save/delete options
- **Chat**: Ask questions, get answers from your knowledge base
- **Email Drafting**: Type "Draft an email about..." → Copy/Send/Draft buttons
- **Streaming**: Responses appear word-by-word in real-time
- **Performance Metrics**: See TTFT, tok/s, and total time on every response

### Header Buttons
- **✨ New**: Start a new conversation
- **📤 Upload**: Upload documents, audio/video, and images
- **🔄 Sync**: Sync knowledge base (green dot = auto-sync enabled)
- **🧠 Memory**: View/delete stored memories
- **🌐 Web**: Toggle web search on/off (green = enabled)
- **⚙️ Settings**: Personality, backup, and more
- All buttons have hover tooltips

### Toggles
- **✍️ Write like me**: Use your personal writing style
- **🧠 Auto-remember**: Automatically save conversations to memory

### Settings (⚙️)
- **AI Personality**: Name and personality prompt
- **Backup & Restore**: Export/import conversations and settings
- **Email Providers**: Gmail OAuth connection
- **Daily Digest**: Schedule and email settings
- **Auto-Sync**: Knowledge base sync interval

## 📊 Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web UI    │────▶│  LLM Server │────▶│ Qwen 2.5 7B │
│  :3000      │     │  :8080      │     │  Q4 GGUF    │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
       ┌───────────┐ ┌─────────┐ ┌─────────┐
       │ ChromaDB  │ │ SQLite  │ │  Gmail  │
       │  (RAG)    │ │ (Chats) │ │  API    │
       │ your docs │ │ Memory  │ │         │
       └───────────┘ └─────────┘ └─────────┘
```

## 🔧 Configuration

Settings are stored in `~/.personal-ai/settings.json` and can be configured via:
- Web UI Settings panel (⚙️ button)
- CLI commands
- Direct file editing

Key settings:
- **AI Personality**: Name and system prompt
- **Daily Digest**: Enable, time, email address
- **Auto-Sync**: Enable, interval (1/4/12/24 hours)
- **Context Window**: 8K/16K/32K tokens

## 📈 Performance & model

**Speed on an RTX 5090 (32 GB), Qwen 2.5 7B Q4_K_M (MEASURED on the v4 fine-tune; speed is set by the architecture, so v5-clean and stock 7B match):**
- **Time to first token:** ~0.03–0.12 s
- **Decode:** 180–215 tok/s sustained (183.9 median); 217.9 on the agentic benchmark
- **Peak VRAM:** ~5.5 GB at 16K context (16K default, 32K max)
- **Datasheet ingestion:** 37 PDFs (6,129 chunks) in ~5 min

**Production model: `skippy-qwen25-7b-v5-clean`** (since 2026-10-14). It is trained only on public datasheet material: grounded answers, explicit "the excerpts don't cover that" abstentions, general knowledge and refusals (`training/build_grounded_dataset.py`, `training/train_lora_v5_clean.py`). The earlier v1–v4 fine-tunes were partly trained on personal email and are retired.

**Why 7B and not 14B:** the 14B scored higher on the accuracy eval (76.2% vs 73.8%) but invented peripherals that don't exist, and it decodes 1.9× slower. Production trades the headline score for calibration. Reasoning: [white paper](docs/skippy-white-paper.md) · [recipe taxonomy](docs/recipe-taxonomy.md).

**Faithfulness detail** (`eval/faithfulness_sweep.py`, 117 answers, temperature 0.7, Sonnet + GPT-4o, MEASURED): fabricated answers 36 (v4) → 22 (stock 7B) → 12 (v5-clean); a repeat v5-clean run gave 16. Net score (Sonnet/GPT-4o): v4 +0.39/+0.28, stock +0.50/+0.47, v5-clean +0.56/+0.56. A v5.1 retrain tied on fabrication but scored lower net (+0.50/+0.52 vs +0.56/+0.55 over 234 answers), so v5-clean stays.

## 🏷️ Versions

| Version | Highlights |
|---------|------------|
| **v5.11.0** | Email-free brain — production model is now `skippy-qwen25-7b-v5-clean`, a fresh Qwen 2.5 7B LoRA trained only on public datasheet material (grounded answers, explicit abstentions, general knowledge, refusals; `training/build_grounded_dataset.py`, `training/train_lora_v5_clean.py`). The email-trained v1–v4 line is retired and archived locally; email chunks removed from search; legacy email auto-retraining disabled. Two-judge 117-answer faithfulness sweep (MEASURED): fabricated answers v4 36 → stock 22 → v5-clean 12. Opt-in HHEM faithfulness-gate sidecar (`faithfulness` service, off by default). Eval scripts read the local login from `~/.personal-ai/keys.env`; personal data scrubbed from repo history. |
| **v5.10.0** | Methodology + research checkpoint — gotcha-#7 closure with N=7 two-factor recipe predictor (ceiling reasoning OR family-match); cross-family v4 campaign (Gemma 9B, Mistral 7B, Llama 3.1 8B, Yi-1.5-9B, Phi-4 14B); Qwen-family substring-grader bias finding + `eval/regrade_semantic.py` semantic-regrade tooling; voice-metrics tooling (`eval/voice_metrics.py`); recipe-taxonomy framework (`docs/recipe-taxonomy.md`); Skippy white paper (`docs/skippy-white-paper.md`); private-anchor secrets spec for cross-app NPU measurement architecture (`docs/private_anchor_secrets_spec.md`); 29-slide use-case deck + presenter script for tech-mgmt audiences; cross-deck reference pattern with Keyhole. Production = Qwen 2.5 7B v4 (73.8% pass / voice ✓ / safety ✓); 14B v4 candidate documented but not shipped (fabrication). No application-feature changes — engine is unchanged. |
| **v5.9.2** | Auth middleware bypasses CORS preflight (`OPTIONS`) — browser was getting 401 on every cross-origin preflight, silently hiding per-user data |
| **v5.9.1** | Multi-user polish — `/health` is public (no more spurious session wipes), header wraps cleanly when buttons don't fit, 401 handler only logs out when a token was actually sent |
| **v5.9.0** | Multi-user support — bcrypt login, per-user data isolation under `users/<name>/`, admin can add/remove users and reset passwords, auto-migrate single-user installs into `users/kyle/` |
| **v5.8.0** | Agent platform + observability — Google Calendar, write-capable tools (sandboxed), reminders, multi-step agent loop, RLHF 👍/👎, training dashboard, selective training exclusion, mobile-friendly UI, Phone Access helper, Prometheus /metrics + Grafana dashboard |
| **v5.7.0** | Smarter RAG, light mode, streaming TTS, training hardening — multi-doc synthesis, conv-aware RAG, citation score breakdown, theme toggle, conversation export |
| **v5.6.0** | Model upgrade — Qwen 2.5 14B replaces Mixtral 8x7B, ChatML format, 32K context, unified training |
| **v5.5.0** | Voice mode — push-to-talk mic input with Whisper STT, browser TTS for spoken responses |
| **v5.4.0** | Agent tools — Skippy can read files, list directories, search web, and check git status (sandboxed, read-only) |
| **v5.3.0** | Multi-turn [INST] format, RAG query rewriting, conversation search, model auto-restore, auto-training schedule, training venv, DuckDuckGo retry |
| **v5.2.0** | Instruct template fix (facts/web search now reliable), integration test suite, training data dedup fix, RAG hybrid search fix |
| **v5.1.0** | Learned Facts layer, inference stability fixes, model warmup, Skippy personality |
| **v5.0.0** | Automated LoRA retraining — Train Now button, GPU handoff, GGUF conversion pipeline |
| **v4.1.0** | Bulk upload (multi-file + folder), enhanced results UI (copy/email/summary), web search toggle, auto-sync indicator, tooltips |
| **v4.0.0** | UI Refactor - Dedicated Upload panel with tabbed Documents/Audio/Images |
| **v3.9.0** | Web Search - DuckDuckGo integration with privacy toggle |
| **v3.8.0** | Meeting Summarization - key points, action items, decisions |
| **v3.7.0** | Image/Screenshot OCR - Tesseract text extraction |
| **v3.6.0** | Audio/Video Transcription - Whisper-powered web upload |
| **v3.5.0** | Document Upload - drag & drop PDFs/TXT/MD in web UI |
| **v3.4.0** | Export/Import - backup & restore conversations and settings |
| **v3.3.1** | Bug fixes: auto-remember, assistant message saving |
| **v3.3.0** | Memory Viewer - browse and delete specific memories |
| **v3.2.0** | AI Personality (custom name/prompt), Performance metrics (TTFT, tok/s) |
| **v3.1.0** | Auto-remember toggle, Clear memory button |
| **v3.0.0** | Conversation Memory - AI remembers past chats! |
| **v2.1.0** | Gmail OAuth, Send/Draft email buttons |
| **v2.0.0** | Streaming, Settings UI, Daily Digest, Datasheet ingestion |
| **v1.0.0** | Initial Release (RAG, email drafting, LoRA training) |

## 📝 License

MIT License - Use freely for personal projects.

## 🙏 Acknowledgments

- [Qwen 2.5](https://qwenlm.github.io/) - Base model (upgraded from Mixtral 8x7B)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI Whisper](https://github.com/openai/whisper) - Audio transcription
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF text extraction

---

**Trust the Awesomeness!** 🚀

## 👤 Maintainer

**Kyle Fox** - [@kylefoxaustin](https://github.com/kylefoxaustin)

Built with ❤️ and lots of 🍺 in Austin, Texas.
