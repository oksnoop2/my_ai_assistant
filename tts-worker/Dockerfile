# tts-worker/Dockerfile (Final Polished Version)

FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set a standard WORKDIR, just like the other services
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    TTS \
    numpy \
    torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
    transformers==4.35.0

EXPOSE 8000
# Use the python file name directly for clarity
CMD ["uvicorn", "tts:app", "--host", "0.0.0.0", "--port", "8000"]
