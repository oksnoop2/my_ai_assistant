# in ./embedding-service/Dockerfile

FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app
COPY . .

RUN pip3 install --cache-dir /cache \
    fastapi \
    uvicorn \
    torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 \
    einops \
    sentence-transformers

EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
