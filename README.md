# 🧠 Personal AI Framework

Your private AI assistant that knows your emails, projects, writing style, and technical documents. Runs 100% locally on your hardware.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Personal Knowledge Base** | 61,000+ documents from emails, transcripts, blogs, datasheets |
| **Your Writing Style** | LoRA fine-tuned on your emails - writes like you |
| **Streaming Responses** | Real-time word-by-word output |
| **Hybrid RAG Search** | BM25 + semantic + reranking for best results |
| **Email Drafting** | Generate emails with one-click open in Gmail/Outlook |
| **Meeting Summarizer** | Transcribe audio/video with Whisper + summarize |
| **Document Generator** | Create specs, proposals, reports from your knowledge |
| **Datasheet Ingestion** | Ingest PDFs - reference manuals, app notes |
| **Daily Digest** | Automated morning summary of your AI activity |
| **Web UI** | Clean interface with settings panel |

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

## 📁 Project Structure
```
personal-ai-framework/
├── docker/                 # Docker configurations
├── knowledge/              # Your knowledge base
│   ├── documents/          # General documents
│   ├── emails/             # Exported emails (.eml, .mbox)
│   ├── transcripts/        # Meeting/video transcripts
│   ├── datasheets/         # PDF datasheets & manuals
│   └── writing/            # Your blog posts, articles
├── pipeline/               # Core Python services
│   ├── llm_server.py       # FastAPI LLM server
│   ├── rag_service.py      # ChromaDB RAG
│   ├── advanced_rag.py     # Hybrid search + reranking
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

- **Chat**: Ask questions, get answers from your knowledge base
- **Email Drafting**: Type "Draft an email about..." → Copy/Save/Open buttons
- **Streaming**: Responses appear word-by-word in real-time
- **Settings** (⚙️): Configure digest, sync, view email provider status

## 📊 Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web UI    │────▶│  LLM Server │────▶│  Mixtral    │
│  :3000      │     │  :8080      │     │  8x7B Q4    │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  ChromaDB   │
                   │  (RAG)      │
                   │  61K+ docs  │
                   └─────────────┘
```

## 🔧 Configuration

Settings are stored in `~/.personal-ai/settings.json` and can be configured via:
- Web UI Settings panel (⚙️ button)
- CLI commands
- Direct file editing

Key settings:
- **Daily Digest**: Enable, time, email address
- **Auto-Sync**: Enable, interval (1/4/12/24 hours)
- **Context Window**: 8K/16K/32K tokens

## 📈 Performance

Tested on RTX 5090 (32GB VRAM):
- **Inference**: ~30 tokens/sec
- **Context**: 16K tokens default (32K max)
- **Knowledge Base**: 61,500 documents
- **Datasheet Ingestion**: 37 PDFs (6,129 chunks) in ~5 min

## 🏷️ Versions

- **v2.0.0** - Advanced Features (streaming, settings UI, daily digest, datasheets)
- **v1.0.0** - Initial Release (RAG, email drafting, LoRA training)

## 📝 License

MIT License - Use freely for personal projects.

## 🙏 Acknowledgments

- [Mixtral 8x7B](https://mistral.ai/) - Base model
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI Whisper](https://github.com/openai/whisper) - Audio transcription

---

**Trust the Awesomeness!** 🚀
