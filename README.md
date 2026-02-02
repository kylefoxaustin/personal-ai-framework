# 🧠 Personal AI Framework

A local, private AI assistant trained on YOUR data - emails, documents, notes, and writing samples. Runs entirely on your hardware with no data leaving your machine.

## Features

- **100% Local & Private** - Your data never leaves your machine
- **RAG-Powered** - Retrieves relevant context from your personal knowledge base
- **54K+ Document Support** - Handles large email archives, PDFs, and more
- **Web Chat Interface** - Easy-to-use browser-based UI
- **Extensible** - Add new document types, customize chunking, fine-tune with LoRA

## Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 3090, 4090, 5090, or similar)
- **RAM**: 32GB+ system memory recommended
- **Storage**: 50GB+ for model and knowledge base
- **OS**: Ubuntu 22.04+ (or similar Linux)
- **Software**: Docker, NVIDIA Container Toolkit

## Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/personal-ai-framework.git
cd personal-ai-framework

# 2. Run setup
./setup.sh

# 3. Add your documents to knowledge/ folder
cp -r ~/my-documents/* knowledge/documents/
cp ~/mail-export.mbox knowledge/emails/

# 4. Start services
./run.sh start

# 5. Ingest your knowledge base
./run.sh ingest

# 6. Open web UI
# Navigate to http://localhost:3000
```

## Directory Structure
```
personal-ai-framework/
├── knowledge/              # YOUR DATA GOES HERE
│   ├── emails/            # Email exports (.mbox, .eml, .pst)
│   ├── documents/         # PDFs, Word docs, text files
│   ├── writing/           # Your writing samples
│   ├── projects/          # Project notes and docs
│   └── transcripts/       # Meeting/video transcripts
├── models/                # AI models (downloaded automatically)
├── pipeline/              # Processing scripts
├── vectordb/              # Vector database storage
├── web/                   # Web chat interface
├── config.yaml            # Configuration
├── setup.sh              # First-time setup
└── run.sh                # Daily operations
```

## Supported File Types

| Type | Extensions | Notes |
|------|------------|-------|
| Text | `.txt`, `.md` | Direct ingestion |
| Email | `.eml`, `.mbox` | Parsed with headers |
| PDF | `.pdf` | Text extraction |
| JSON | `.json` | Content field extraction |
| Outlook | `.pst` | Requires extraction step |

## Commands
```bash
./run.sh start           # Start all services
./run.sh stop            # Stop all services
./run.sh status          # Check service health
./run.sh ingest          # Ingest knowledge base
./run.sh ingest-emails   # Ingest extracted PST emails
./run.sh logs            # View server logs
./run.sh clear-knowledge # Reset knowledge base
```

## Extracting PST Files

Outlook PST files need to be extracted before ingestion:
```bash
# Install pst-utils
sudo apt install pst-utils

# Extract PSTs
cd knowledge/emails
python3 ../../pipeline/extract_psts.py

# Ingest extracted emails
cd ../..
./run.sh ingest-emails --sent-only  # Start with sent items
./run.sh ingest-emails              # Then all emails
```

## Configuration

Edit `config.yaml` to customize:

- Model settings (context length, temperature)
- Chunking parameters
- File type filters
- Server ports

## Hardware Tested

| GPU | VRAM | Performance |
|-----|------|-------------|
| RTX 5090 | 32GB | ~147 tok/s |
| RTX 4090 | 24GB | ~100 tok/s |
| RTX 3090 | 24GB | ~60 tok/s |

## Troubleshooting

**"NVIDIA driver mismatch"**
```bash
sudo reboot  # Reload NVIDIA modules
```

**"Connection refused" on web UI**
```bash
./run.sh status  # Check if services are running
./run.sh restart # Restart services
```

**Slow first query**
- First query after start caches the model - subsequent queries are faster

## License

MIT License - See LICENSE file

## Acknowledgments

- Mixtral 8x7B by Mistral AI
- llama.cpp for efficient inference
- ChromaDB for vector storage
