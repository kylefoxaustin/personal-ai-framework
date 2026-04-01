# 🧠 Personal AI Framework

Your private AI assistant that knows your emails, projects, writing style, and **remembers your conversations**. Runs 100% locally on your hardware.

![Version](https://img.shields.io/badge/version-4.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

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
| **Personal Knowledge Base** | 61,000+ documents from emails, transcripts, blogs, datasheets |
| **Your Writing Style** | LoRA fine-tuned on your emails - writes like you |
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
- **OS**: Ubuntu 22.04+ (or similar Linux)

## 🚀 Quick Start
```bash
# Clone the repo
git clone https://github.com/kylefoxaustin/personal-ai-framework.git
cd personal-ai-framework

# Run setup (installs dependencies, creates config)
./setup.sh

# Download model (Mixtral 8x7B Q4)
./run.sh download-model

# Start services
./run.sh start

# Open web UI
xdg-open http://localhost:3000
```

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
│   Web UI    │────▶│  LLM Server │────▶│  Mixtral    │
│  :3000      │     │  :8080      │     │  8x7B Q4    │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
       ┌───────────┐ ┌─────────┐ ┌─────────┐
       │ ChromaDB  │ │ SQLite  │ │  Gmail  │
       │  (RAG)    │ │ (Chats) │ │  API    │
       │ 61K docs  │ │ Memory  │ │         │
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

## 📈 Performance

Tested on RTX 5090 (32GB VRAM):
- **TTFT**: ~0.04-0.17s (time to first token)
- **Throughput**: 85-140 tokens/sec
- **Context**: 16K tokens default (32K max)
- **Knowledge Base**: 61,500+ documents
- **Datasheet Ingestion**: 37 PDFs (6,129 chunks) in ~5 min

## 🏷️ Versions

| Version | Highlights |
|---------|------------|
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

- [Mixtral 8x7B](https://mistral.ai/) - Base model
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI Whisper](https://github.com/openai/whisper) - Audio transcription
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF text extraction

---

**Trust the Awesomeness!** 🚀

## 👤 Maintainer

**Kyle Fox** - [@kylefoxaustin](https://github.com/kylefoxaustin)

Built with ❤️ and lots of 🍺 in Austin, Texas.
