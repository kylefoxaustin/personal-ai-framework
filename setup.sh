#!/bin/bash
# =============================================================================
# Personal AI Framework - Setup Script
# =============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=============================================="
echo "🧠 Personal AI Framework Setup"
echo "=============================================="
echo ""

# Check for NVIDIA GPU
echo "Checking prerequisites..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected. This framework requires an NVIDIA GPU.${NC}"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo -e "${GREEN}✅ GPU: $GPU_NAME ($GPU_MEM)${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not installed. Please install Docker first.${NC}"
    echo "   Visit: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

# Check for NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -qi "nvidia\|runtimes"; then
    echo -e "${YELLOW}⚠️  NVIDIA Container Toolkit may not be configured.${NC}"
    echo "   Install with:"
    echo "   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "   sudo nvidia-ctk runtime configure --runtime=docker"
    echo "   sudo systemctl restart docker"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ NVIDIA Container Toolkit configured${NC}"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not installed.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3 installed${NC}"

echo ""
echo "📁 Creating directory structure..."
mkdir -p knowledge/{emails,projects,transcripts,writing,documents}
mkdir -p knowledge/emails/extracted
mkdir -p models
mkdir -p vectordb
mkdir -p training/{data,output}

# Create .gitkeep files to preserve structure
touch knowledge/.gitkeep
touch knowledge/emails/.gitkeep
touch knowledge/documents/.gitkeep
touch knowledge/writing/.gitkeep
touch knowledge/projects/.gitkeep
touch knowledge/transcripts/.gitkeep

echo -e "${GREEN}✅ Directories created${NC}"

# Setup config file
if [ ! -f "pipeline/config.yaml" ]; then
    echo "📄 Creating config from template..."
    cp pipeline/config.yaml.example pipeline/config.yaml
    echo -e "${GREEN}✅ Config created - edit pipeline/config.yaml to customize${NC}"
else
    echo -e "${GREEN}✅ Config file exists${NC}"
fi

# Install Python dependencies for training (optional)
echo ""
read -p "Install Python dependencies for training/development? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing Python packages..."
    pip3 install --quiet huggingface_hub watchdog pyyaml requests
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
fi

# Model selection
echo ""
echo "=============================================="
echo "📥 Model Selection"
echo "=============================================="
echo ""
echo "Choose a base model:"
echo -e "${BLUE}1)${NC} Mistral 7B Instruct (~4.4GB) - Fast, good for most uses"
echo -e "${BLUE}2)${NC} Mixtral 8x7B Instruct (~26GB) - More capable, needs 24GB+ VRAM"
echo -e "${BLUE}3)${NC} Skip - I'll provide my own model"
echo ""
read -p "Select [1/2/3]: " -n 1 -r MODEL_CHOICE
echo ""

case $MODEL_CHOICE in
    1)
        MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        MODEL_FILE="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        MODEL_SIZE="4.4GB"
        ;;
    2)
        MODEL_URL="https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
        MODEL_FILE="models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
        MODEL_SIZE="26GB"
        ;;
    3)
        echo "Skipping model download."
        echo -e "${YELLOW}⚠️  Update pipeline/config.yaml with your model path${NC}"
        MODEL_FILE=""
        ;;
    *)
        echo "Invalid choice, skipping model download."
        MODEL_FILE=""
        ;;
esac

if [ -n "$MODEL_FILE" ]; then
    if [ -f "$MODEL_FILE" ]; then
        echo -e "${GREEN}✅ Model already exists: $MODEL_FILE${NC}"
    else
        echo "📥 Downloading model (~$MODEL_SIZE)..."
        echo "   This may take a while depending on your connection."
        wget -c "$MODEL_URL" -O "$MODEL_FILE" --show-progress
        echo -e "${GREEN}✅ Model downloaded${NC}"
        
        # Update config with correct model path
        if [ -f "pipeline/config.yaml" ]; then
            sed -i "s|path:.*\.gguf|path: \"/app/$(basename $MODEL_FILE)\"|" pipeline/config.yaml
        fi
    fi
fi

# Build Docker images
echo ""
echo "🐳 Building Docker images..."
docker compose build

echo ""
echo "=============================================="
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "=============================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1. ${BLUE}Add your documents:${NC}"
echo "   knowledge/emails/      - Email exports (.eml, .mbox)"
echo "   knowledge/documents/   - PDFs, Word docs, text files"
echo "   knowledge/writing/     - Your blog posts, articles"
echo "   knowledge/projects/    - Code projects, READMEs"
echo ""
echo "   For Outlook PST files:"
echo "   sudo apt install pst-utils"
echo "   readpst -o knowledge/emails/extracted -e your_archive.pst"
echo ""
echo "2. ${BLUE}Start the services:${NC}"
echo "   ./run.sh start"
echo ""
echo "3. ${BLUE}Ingest your documents:${NC}"
echo "   ./run.sh ingest"
echo ""
echo "4. ${BLUE}Open the web UI:${NC}"
echo "   cd web && python3 -m http.server 8765"
echo "   Then visit: http://localhost:8765"
echo ""
echo "5. ${BLUE}(Optional) Train on your writing style:${NC}"
echo "   See docs/FRESH_INSTALL_GUIDE.md"
echo ""
echo "🚀 Trust the Awesomeness!"
