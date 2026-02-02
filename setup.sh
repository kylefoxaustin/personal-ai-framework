#!/bin/bash
# =============================================================================
# Personal AI Framework - Setup Script
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "🧠 Personal AI Framework Setup"
echo "=============================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA GPU not detected. This framework requires an NVIDIA GPU.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not installed. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

# Check for NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}⚠️  NVIDIA Container Toolkit may not be configured.${NC}"
    echo "   Run: sudo apt install nvidia-container-toolkit"
fi

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p knowledge/{emails,projects,transcripts,writing,documents}
mkdir -p models/mixtral
mkdir -p vectordb

echo -e "${GREEN}✅ Directories created${NC}"

# Check for model
MODEL_PATH="models/mixtral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Mixtral model not found at $MODEL_PATH${NC}"
    echo ""
    read -p "Download Mixtral 8x7B Q4_K_M (~26GB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📥 Downloading model (this will take a while)..."
        wget -c https://huggingface.co/mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf \
            -O "$MODEL_PATH"
        echo -e "${GREEN}✅ Model downloaded${NC}"
    else
        echo "Skipping model download. You'll need to download it manually."
    fi
else
    echo -e "${GREEN}✅ Model found${NC}"
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
echo "Next steps:"
echo "1. Add your documents to the knowledge/ folder:"
echo "   - knowledge/emails/      - Email exports (.mbox, .eml, .pst)"
echo "   - knowledge/documents/   - Documents (.pdf, .docx, .txt)"
echo "   - knowledge/writing/     - Your writing samples"
echo "   - knowledge/projects/    - Project notes"
echo ""
echo "2. Start the services:"
echo "   ./run.sh start"
echo ""
echo "3. Ingest your knowledge base:"
echo "   ./run.sh ingest"
echo ""
echo "4. Open the web UI:"
echo "   http://localhost:3000"
echo ""
