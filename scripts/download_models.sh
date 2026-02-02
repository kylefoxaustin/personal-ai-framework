#!/bin/bash

echo "======================================="
echo "Personal AI Framework - Model Setup"
echo "======================================="

# Create model directory
mkdir -p models/mixtral

cd models/mixtral

# Download Mixtral 8x7B Instruct (4-bit quantized for your RTX 5090)
echo "Downloading Mixtral 8x7B Instruct (Q4_K_M - about 23GB)..."
wget https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf

# Download the config files
echo "Downloading configuration files..."
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/config.json
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/tokenizer.json
wget https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/raw/main/tokenizer_config.json

echo "✅ Model download complete!"
ls -lh *.gguf
