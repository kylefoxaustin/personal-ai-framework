# Personal AI Framework - Fresh Install Guide

A complete guide to setting up your own personal AI that knows your work, writes like you, and runs 100% locally.

## What You'll Get

- 🧠 AI assistant trained on YOUR documents and writing style
- 🔒 100% local - no cloud, no data sharing
- 💬 Web chat interface with conversation memory
- 🔄 Smart sync that auto-updates when you add files
- ✍️ "Write like me" mode for emails in your voice

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 12GB VRAM | NVIDIA 24GB+ VRAM |
| RAM | 32GB | 64GB |
| Storage | 100GB free | 250GB+ free |
| CPU | 8 cores | 16+ cores |

**Tested GPUs:** RTX 3090, RTX 4090, RTX 5090, A100

### Software Requirements

- Ubuntu 22.04+ (or WSL2 on Windows)
- Docker with NVIDIA Container Toolkit
- Python 3.10+
- NVIDIA Driver 525+

---

## Step 1: System Setup

### Install NVIDIA Container Toolkit
```bash
# Add NVIDIA repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Install Python Dependencies
```bash
sudo apt-get install -y python3-pip python3-venv
pip3 install huggingface_hub
```

---

## Step 2: Clone and Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/personal-ai-framework.git
cd personal-ai-framework

# Run setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Check GPU availability
- Create directory structure
- Download the base model (~15GB)
- Build Docker images

---

## Step 3: Add Your Data

### Directory Structure
```
knowledge/
├── emails/           # Your email exports
│   └── extracted/    # Extracted PST/MBOX files
├── documents/        # PDFs, Word docs, text files
├── writing/          # Blog posts, articles you've written
├── projects/         # Code projects, READMEs
└── transcripts/      # Meeting notes, podcast transcripts
```

### Supported File Types

| Type | Extensions |
|------|------------|
| Text | .txt, .md, .rst |
| Documents | .pdf, .docx, .doc |
| Email | .eml, .mbox |
| Data | .json, .csv |
| Code | .py, .js, .cpp, etc. |

### Adding Emails (Outlook PST)

If you have Outlook PST files:
```bash
# Install readpst
sudo apt-get install -y pst-utils

# Extract PST to individual emails
mkdir -p knowledge/emails/extracted
readpst -o knowledge/emails/extracted -e your_archive.pst

# This creates folders like:
# knowledge/emails/extracted/Inbox/
# knowledge/emails/extracted/Sent Items/
```

### Adding Emails (Gmail/MBOX)

1. Go to Google Takeout: https://takeout.google.com
2. Select only "Mail" 
3. Download the MBOX file
4. Place in `knowledge/emails/`

### Adding Documents

Simply copy files into the appropriate folders:
```bash
cp ~/Documents/work/*.pdf knowledge/documents/
cp ~/Blog/posts/*.md knowledge/writing/
```

---

## Step 4: Start Services & Ingest

### Start the AI Server
```bash
./run.sh start
```

Wait ~30 seconds for the model to load, then verify:
```bash
curl http://localhost:8080/health
# Should show: {"status":"healthy","model_loaded":true,...}
```

### Ingest Your Documents
```bash
# Ingest all documents
./run.sh ingest

# Or ingest just emails
./run.sh ingest-emails
```

This will:
- Scan all files in `knowledge/`
- Chunk them intelligently (respecting paragraphs, email structure)
- Add to the vector database

**Time estimate:** ~1 minute per 1,000 documents

### Verify Ingestion
```bash
curl http://localhost:8080/health
# Check "knowledge_base_documents" count
```

---

## Step 5: Test Your AI

### Open the Web Interface
```bash
cd web
python3 -m http.server 3000
```

Open http://localhost:3000 in your browser.

### Try Some Queries

- "What projects have I worked on?"
- "Summarize my recent emails about [topic]"
- "What do I know about [person/company]?"

---

## Step 6: Train on Your Writing Style (Optional)

This step creates a personalized model that writes like YOU.

### Prerequisites

- At least 5,000+ sent emails recommended
- ~5GB additional disk space
- 3-4 hours training time

### Prepare Training Data
```bash
# This extracts your sent emails into training format
python3 training/prepare_training_data.py
```

**Output:**
```
Total emails processed: 20,000
Instruction pairs: 12,000
  - Training: 10,800
  - Validation: 1,200
```

### Configure Exclusions (Optional)

Edit `training/prepare_training_data.py` to exclude certain senders:
```python
EXCLUDE_SENDERS = [
    "spouse_name",
    "family_member",
    # Add emails you don't want the AI to learn from
]
```

### Run Training
```bash
# Stop the inference server to free GPU memory
docker compose stop llm-server

# Start training (3-5 hours)
python3 training/train_lora.py 2>&1 | tee training/training.log

# Monitor progress
tail -f training/training.log
```

### Merge and Deploy

After training completes:
```bash
# Merge LoRA adapters into base model
python3 training/merge_lora.py

# Install llama.cpp for quantization
cd ~/Documents/GitHub
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
make -j$(nproc) llama-quantize

# Convert to GGUF
python3 ../convert_hf_to_gguf.py \
  ~/Documents/GitHub/personal-ai-framework/training/output/merged \
  --outfile ~/Documents/GitHub/personal-ai-framework/models/personal-7b-f16.gguf \
  --outtype f16

# Quantize for faster inference
./bin/llama-quantize \
  ~/Documents/GitHub/personal-ai-framework/models/personal-7b-f16.gguf \
  ~/Documents/GitHub/personal-ai-framework/models/personal-7b-q4_k_m.gguf \
  Q4_K_M

# Update config to use new model
sed -i 's|path:.*\.gguf|path: "/app/models/personal-7b-q4_k_m.gguf"|' pipeline/config.yaml

# Restart server
docker compose start llm-server
```

---

## Step 7: Daily Usage

### Commands Reference

| Command | Description |
|---------|-------------|
| `./run.sh start` | Start all services |
| `./run.sh stop` | Stop all services |
| `./run.sh status` | Check service status |
| `./run.sh ingest` | Ingest new documents |
| `./run.sh sync-now` | Force sync changes |
| `./run.sh watch` | Auto-sync file changes |
| `./run.sh logs` | View server logs |

### Web Interface Features

- **New Chat** - Clear conversation history
- **Sync** - Update knowledge base with new files
- **Write like me** - Toggle personal writing style

### Adding New Documents

1. Drop files in the appropriate `knowledge/` folder
2. Click "Sync" in the web UI, or run `./run.sh sync-now`

---

## Troubleshooting

### "Model not loaded" Error
```bash
# Check if container is running
docker compose ps

# Check logs
docker compose logs llm-server --tail 50

# Restart
docker compose restart llm-server
```

### Out of GPU Memory
```bash
# Check what's using GPU
nvidia-smi

# Stop server before training
docker compose stop llm-server
```

### Slow Ingestion

- Reduce chunk size in `config.yaml`
- Process in batches: `./run.sh ingest --batch 1000`

### RAG Not Finding Relevant Docs

- Check document was ingested: search for specific text
- Verify file extension is supported
- Check `knowledge/` folder permissions

---

## Customization

### Change Model

Edit `pipeline/config.yaml`:
```yaml
model:
  path: "/app/models/your-model.gguf"
```

Supported models: Any GGUF-format model (Mistral, Llama, etc.)

### Adjust Chunk Size

Edit `pipeline/config.yaml`:
```yaml
knowledge_base:
  chunk_size: 500      # Words per chunk
  chunk_overlap: 50    # Overlap between chunks
```

### Change Training Parameters

Edit `training/train_lora.py`:
```python
LORA_R = 64           # LoRA rank (higher = more capacity)
EPOCHS = 3            # Training epochs
LEARNING_RATE = 2e-4  # Learning rate
```

---

## Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface (:3000)                │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  LLM Server (:8080)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Mistral   │  │     RAG     │  │  Smart Chunker  │ │
│  │   7B/8x7B   │  │   Service   │  │                 │ │
│  └─────────────┘  └──────┬──────┘  └─────────────────┘ │
└──────────────────────────┼──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  ChromaDB (:8000)                       │
│              Vector Database for RAG                    │
└─────────────────────────────────────────────────────────┘
```

---

## Security Notes

- All data stays local - nothing leaves your machine
- No API keys or cloud services required
- Knowledge base is stored in `vectordb/` (gitignored)
- Training data is stored in `training/data/` (gitignored)

---

## Getting Help

- Check logs: `./run.sh logs`
- GPU status: `nvidia-smi`
- Service status: `docker compose ps`

---

## Credits

Built with:
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [PEFT](https://github.com/huggingface/peft) - LoRA fine-tuning
- [Mistral](https://mistral.ai/) - Base model

---

**Trust the Awesomeness!** 🚀
