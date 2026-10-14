# Faithfulness gate sidecar: HHEM-2.1-Open on CPU. Kept out of the llm-server image so that
# image stays CUDA-only and the gate can never contend for the GPU.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==4.57.6 sentencepiece protobuf fastapi uvicorn
WORKDIR /app
CMD ["uvicorn", "faithfulness_service:app", "--app-dir", "/app/pipeline", "--host", "0.0.0.0", "--port", "8090"]
