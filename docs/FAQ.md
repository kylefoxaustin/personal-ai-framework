# Frequently Asked Questions

## Setup & Installation

### Q: What GPU do I need?
**Minimum:** 12GB VRAM (RTX 3080, 3090, 4070 Ti)
**Recommended:** 24GB+ VRAM (RTX 3090, 4090, 5090, A100)

The base model runs on 12GB. Training requires 24GB+.

### Q: Can I run this on CPU only?
Not recommended. Inference would be extremely slow (minutes per response). A GPU is essential for usable performance.

### Q: Does this work on Windows?
Yes, via WSL2 with Ubuntu. Native Windows is not supported.

### Q: How much disk space do I need?
- Base install: ~20GB
- With training: ~50GB
- Full setup with cached models: ~100GB

---

## Data & Privacy

### Q: Where is my data stored?
Everything stays local:
- Documents: `knowledge/` folder
- Vector DB: `vectordb/` folder
- Training data: `training/data/` folder

Nothing leaves your machine. No cloud services.

### Q: Can I delete specific documents from the AI's memory?
Yes! Use the sync feature:
1. Delete the file from `knowledge/`
2. Click "Sync" in web UI or run `./run.sh sync-now`
3. Old chunks are automatically removed

### Q: What file types are supported?
Text: `.txt`, `.md`, `.rst`
Documents: `.pdf`, `.docx`, `.doc`
Email: `.eml`, `.mbox`, `.pst` (needs extraction)
Data: `.json`, `.csv`
Code: `.py`, `.js`, `.cpp`, etc.

---

## Training

### Q: How long does training take?
| Examples | Time (RTX 4090) | Time (RTX 5090) |
|----------|-----------------|-----------------|
| 5,000 | ~1 hour | ~45 min |
| 15,000 | ~3 hours | ~2 hours |
| 25,000 | ~5 hours | ~3.5 hours |

### Q: How many emails do I need for good results?
- Minimum: 1,000 sent emails
- Good: 5,000+ sent emails
- Best: 10,000+ sent emails

More data = better style matching.

### Q: Can I exclude certain emails from training?
Yes! Edit `training/prepare_training_data.py`:
```python
EXCLUDE_SENDERS = [
    "spouse@email.com",
    "family",
    # Add more to exclude
]
```

### Q: Do I need to retrain if I add new documents?
No! New documents are added via ingestion (RAG), not training.

Training is only for writing style. RAG handles knowledge.

---

## Usage

### Q: Why isn't the AI finding my documents?
1. Check the file was ingested: `curl http://localhost:8080/health`
2. Verify file type is supported
3. Try resyncing: `./run.sh sync-now`
4. Check file isn't empty or corrupted

### Q: How do I update the knowledge base?
1. Add files to `knowledge/` folder
2. Click "Sync" in web UI, or:
```bash
   ./run.sh sync-now
```

### Q: What does "Write like me" toggle do?
When ON: Uses your trained writing style for emails
When OFF: Uses generic assistant style

Only affects writing tasks (emails, drafts). Regular Q&A is unaffected.

### Q: Can I use a different base model?
Yes! Any GGUF-format model works. Edit `pipeline/config.yaml`:
```yaml
model:
  path: "/app/models/your-model.gguf"
```

Popular options: Llama 3, Mistral, Mixtral, Qwen

---

## Troubleshooting

### Q: "Connection refused" error
```bash
# Check if services are running
docker compose ps

# Restart if needed
./run.sh restart
```

### Q: "Out of memory" error
```bash
# Check GPU usage
nvidia-smi

# Stop server before training
docker compose stop llm-server
```

### Q: Slow responses
- Check GPU is being used: `nvidia-smi` during inference
- Reduce `max_tokens` in request
- Use quantized model (Q4_K_M)

### Q: Training crashed
- Check GPU memory was free before starting
- Look at logs: `tail -100 training/training.log`
- Reduce batch size in `train_lora.py`

### Q: Model outputs garbage
- Model may not have loaded correctly
- Check logs: `docker compose logs llm-server`
- Restart: `docker compose restart llm-server`

---

## Performance

### Q: What speeds should I expect?

| GPU | Inference Speed | Training Speed |
|-----|-----------------|----------------|
| RTX 3090 | ~80 tok/s | ~3.5s/step |
| RTX 4090 | ~120 tok/s | ~2.5s/step |
| RTX 5090 | ~150 tok/s | ~2.0s/step |

### Q: How can I make it faster?
1. Use quantized models (Q4_K_M)
2. Reduce context length in config
3. Close other GPU applications
4. Use a faster GPU 😄

---

## Still stuck?

1. Check logs: `./run.sh logs`
2. GPU status: `nvidia-smi`
3. Service health: `curl http://localhost:8080/health`
4. Full restart: `./run.sh stop && ./run.sh start`
