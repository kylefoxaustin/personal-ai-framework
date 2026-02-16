# Quick Start (TL;DR)

Get your personal AI running in 15 minutes.

## Prerequisites
- NVIDIA GPU (12GB+ VRAM)
- Docker with nvidia-container-toolkit
- Ubuntu 22.04+ or WSL2

## Setup
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/personal-ai-framework.git
cd personal-ai-framework

# Setup (downloads model, builds containers)
./setup.sh

# Add your documents
cp ~/Documents/*.pdf knowledge/documents/
cp ~/emails/*.eml knowledge/emails/

# Start
./run.sh start

# Wait 30 seconds, then ingest
./run.sh ingest

# Open web UI
cd web && python3 -m http.server 3000
# Go to http://localhost:3000
```

## That's It!

Ask your AI:
- "What do I know about [topic]?"
- "Summarize my emails about [project]"
- "Write an email about [subject]"

## Want It to Write Like You?
```bash
# Prepare training data from sent emails
python3 training/prepare_training_data.py

# Train (~3 hours)
docker compose stop llm-server
python3 training/train_lora.py

# Deploy (after training completes)
python3 training/merge_lora.py
# Follow docs/FRESH_INSTALL_GUIDE.md for quantization steps
```

## Commands

| Command | What it does |
|---------|--------------|
| `./run.sh start` | Start AI |
| `./run.sh stop` | Stop AI |
| `./run.sh ingest` | Add new docs |
| `./run.sh sync-now` | Quick sync |
| `./run.sh logs` | Debug |

📚 Full guide: [docs/FRESH_INSTALL_GUIDE.md](FRESH_INSTALL_GUIDE.md)
