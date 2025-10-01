# in ./rag-service/Dockerfile

# Start from the same CUDA runtime as other GPU services
FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install the required python libraries, targeting CUDA 12.4
RUN python3 -m pip install --no-cache-dir \
    fastapi \
    uvicorn \
    watchdog \
    neo4j \
    python-dotenv \
    llama-index-core \
    llama-index-llms-llama-cpp \
    llama-index-embeddings-huggingface \
    llama-index-graph-stores-neo4j \
    llama-index-readers-file \
    einops \
    torch --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
