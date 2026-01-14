# Personal AI Framework - System Architecture

## Overview
A modular system combining large language models with personal knowledge and style adaptation.

## Components

### 1. Base Model Layer
- **Primary**: Mixtral 8x7B (47B MoE model, ~13B active)
- **Alternative**: Llama 3.1 70B (4-bit quantized)
- **Inference**: llama.cpp with CUDA acceleration
- **Memory**: ~20-25GB VRAM for Mixtral, ~35GB for Llama 70B

### 2. Personalization Layer
- **Method**: LoRA adapters (low-rank adaptation)
- **Training Data**: Author's complete written works
- **Update Schedule**: Weekly or on-demand
- **Memory**: +500MB-1GB per adapter

### 3. Knowledge Base
- **Vector DB**: ChromaDB or Qdrant
- **Embeddings**: BGE-large or E5-large
- **Updates**: Nightly ingestion of new content
- **Content Types**: Emails, documents, projects, notes

### 4. Pipeline Architecture
```
User Query → Context Extraction → RAG Retrieval → LLM Generation → Style Transfer → Response
```

### 5. Interfaces
- Web UI (Flask/FastAPI)
- CLI tools
- REST API for integrations

## Hardware Utilization
- GPU: Model inference (20-35GB VRAM)
- CPU: Embedding generation, vector search
- RAM: Document cache, context window (10-20GB)
- Storage: Vector DB, model files (~100GB)
