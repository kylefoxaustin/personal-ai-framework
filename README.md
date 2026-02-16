# 🧠 Personal AI Framework

A private, local AI assistant that knows your work and writes like you.

![Status](https://img.shields.io/badge/status-working-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Features

- **📚 Knowledge Base** - Ingest your emails, documents, and notes
- **🔍 Smart Search** - RAG-powered retrieval finds relevant context
- **✍️ Your Voice** - Fine-tune to write emails in your style
- **💬 Chat Memory** - Multi-turn conversations with context
- **🔄 Auto Sync** - Detects file changes automatically
- **🔒 100% Local** - No cloud, no data sharing, no API keys

## 🚀 Quick Start
```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/personal-ai-framework.git
cd personal-ai-framework
./setup.sh

# Add your documents
cp ~/Documents/*.pdf knowledge/documents/
cp ~/emails/*.eml knowledge/emails/

# Start and ingest
./run.sh start
./run.sh ingest

# Open web UI
cd web && python3 -m http.server 3000
# Visit http://localhost:3000
```

## 📋 Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA 12GB+ VRAM |
| RAM | 32GB+ |
| OS | Ubuntu 22.04+ / WSL2 |
| Docker | With nvidia-container-toolkit |

## 📁 Project Structure
```
personal-ai-framework/
├── knowledge/          # Your documents (gitignored)
│   ├── emails/         # Email exports
│   ├── documents/      # PDFs, docs
│   ├── writing/        # Your blog posts, articles
│   └── projects/       # Code, READMEs
├── models/             # AI models
├── pipeline/           # Core services
├── training/           # LoRA fine-tuning
├── web/                # Chat interface
├── config.yaml         # Configuration
├── setup.sh            # First-time setup
└── run.sh              # Daily operations
```

## 🎯 Commands

| Command | Description |
|---------|-------------|
| `./run.sh start` | Start all services |
| `./run.sh stop` | Stop all services |
| `./run.sh ingest` | Add documents to knowledge base |
| `./run.sh sync-now` | Sync file changes |
| `./run.sh watch` | Auto-sync on file changes |
| `./run.sh logs` | View server logs |
| `./run.sh status` | Check service health |

## 🎨 Web Interface

Access at `http://localhost:3000` after starting services.

Features:
- **New Chat** - Start fresh conversation
- **Sync** - Update knowledge base
- **Write Like Me** - Toggle your personal writing style

## 📖 Documentation

- **[Quick Start](docs/QUICKSTART.md)** - Get running in 15 minutes
- **[Full Install Guide](docs/FRESH_INSTALL_GUIDE.md)** - Detailed setup instructions
- **[FAQ](docs/FAQ.md)** - Common questions answered
- **[Architecture](docs/ARCHITECTURE.md)** - Technical details

## 🔧 Training Your Writing Style

Make the AI write like you:
```bash
# Prepare data from sent emails
python3 training/prepare_training_data.py

# Train (~3 hours on RTX 4090)
docker compose stop llm-server
python3 training/train_lora.py

# Deploy trained model
python3 training/merge_lora.py
# See docs for quantization steps
```

## 📊 Performance

| GPU | Inference | Training (10K examples) |
|-----|-----------|-------------------------|
| RTX 3090 | ~80 tok/s | ~4 hours |
| RTX 4090 | ~120 tok/s | ~2.5 hours |
| RTX 5090 | ~150 tok/s | ~2 hours |

## 🔒 Privacy

- All data stays on your machine
- No cloud services or API keys required
- Models run 100% locally
- Knowledge base is gitignored

## 🙏 Built With

- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [PEFT](https://github.com/huggingface/peft) - LoRA fine-tuning
- [Mistral](https://mistral.ai/) - Base model

## 📄 License

MIT License - use it however you want!

---

**Trust the Awesomeness!** 🚀
